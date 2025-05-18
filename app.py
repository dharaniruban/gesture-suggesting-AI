import cv2
import mediapipe as mp
import numpy as np
import time
import os
import requests
import re
import moviepy.editor as mp_editor
import whisper
import textwrap
import json
import logging  

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils

# --- Hand Movement Detection ---
def process_video_for_hands(input_path):
    if not os.path.exists(input_path):
        print(f"Error: Video file '{input_path}' not found.")
        return False, 0, []

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video '{input_path}'.")
        return False, 0, []

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    frame_time = 1000 / fps

    hand_outside_detected = False
    start_time = time.time()
    hand_moved_times = []
    frame_count = 0

    window_name = "Input vs Processed Video"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    left_wrist_history = []
    right_wrist_history = []
    history_size = 5
    movement_threshold = 0.06 * height

    while cap.isOpened():
        frame_start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        input_frame = frame.copy()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = True
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        hands_outside = False
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            eye_y = int(landmarks[5].y * height)
            hip_left = landmarks[23]
            hip_right = landmarks[24]
            belly_y = int((hip_left.y + hip_right.y) * height / 2) - int(0.02 * height)

            cv2.line(image, (0, eye_y), (width, eye_y), (0, 0, 255), 2)
            cv2.line(image, (0, belly_y), (width, belly_y), (0, 0, 255), 2)

            left_wrist_y = landmarks[15].y * height
            right_wrist_y = landmarks[16].y * height

            left_wrist_history.append(left_wrist_y)
            right_wrist_history.append(right_wrist_y)
            if len(left_wrist_history) > history_size:
                left_wrist_history.pop(0)
                right_wrist_history.pop(0)

            outside_lines = (left_wrist_y < eye_y or left_wrist_y > belly_y) or \
                           (right_wrist_y < eye_y or right_wrist_y > belly_y)

            noticeable_movement = False
            if len(left_wrist_history) == history_size:
                left_max_diff = max(left_wrist_history) - min(left_wrist_history)
                right_max_diff = max(right_wrist_history) - min(right_wrist_history)
                if left_max_diff > movement_threshold or right_max_diff > movement_threshold:
                    noticeable_movement = True

            if outside_lines and noticeable_movement:
                hands_outside = True
                hand_outside_detected = True
                timestamp = round(frame_count / fps)
                if timestamp not in hand_moved_times:
                    hand_moved_times.append(timestamp)
                cv2.putText(image, "Hands Moved", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                                      connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2))

        parallel_frame = np.hstack((input_frame, image))
        cv2.imshow(window_name, parallel_frame)

        frame_processing_time = (time.time() - frame_start_time) * 1000
        wait_time = max(1, int(frame_time - frame_processing_time))
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

        frame_count += 1

    processing_time = time.time() - start_time
    cap.release()
    cv2.destroyAllWindows()
    
    return hand_outside_detected, processing_time, sorted(hand_moved_times)

# --- Speech Transcription and Intent Classification ---
def extract_audio_from_video(video_path, audio_path="temp_audio.wav"):
    try:
        video = mp_editor.VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path)
        video.close()
        print(f"Audio extracted successfully to {audio_path}")
        return audio_path
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None

def recognize_speech_with_whisper(audio_path, language="en"):
    try:
        model = whisper.load_model("tiny")
        print("Recognizing speech with Whisper...")
        result = model.transcribe(audio_path, language=language)
        segments = result["segments"]
        
        sentences_map = {}
        for segment in segments:
            start_time = segment["start"]
            end_time = segment["end"]
            text = segment["text"].strip()
            segment_sentences = re.split(r'(?<=[.!?])\s+', text)
            for sentence in segment_sentences:
                if sentence:
                    sentences_map[sentence] = {"start": start_time, "end": end_time}
        
        full_text = result["text"].strip()
        return full_text, sentences_map
    except Exception as e:
        return f"Error processing audio: {e}", {}

def find_best_matching_timestamp(sentence, sentences_map):
    if sentence in sentences_map:
        return sentences_map[sentence]
    
    best_match = None
    highest_similarity = 0
    clean_sentence = re.sub(r'[^\w\s]', '', sentence.lower())
    
    for s, timestamp in sentences_map.items():
        clean_s = re.sub(r'[^\w\s]', '', s.lower())
        words1 = set(clean_sentence.split())
        words2 = set(clean_s.split())
        
        if not words1 or not words2:
            continue
            
        overlap = len(words1.intersection(words2))
        total = len(words1.union(words2))
        similarity = overlap / total if total > 0 else 0
        
        if similarity > highest_similarity and similarity > 0.5:
            highest_similarity = similarity
            best_match = timestamp
    
    return best_match or {"start": 0, "end": 0}

def format_time(seconds):
    minutes = int(seconds // 60)
    seconds = seconds % 60
    return f"{minutes:02d}:{seconds:05.2f}"

def transcribe_video_speech(video_path, language="en"):
    audio_path = "temp_audio.wav"
    extracted_audio = extract_audio_from_video(video_path, audio_path)
    if not extracted_audio:
        return "Failed to extract audio.", {}
    
    full_text, sentences_map = recognize_speech_with_whisper(audio_path, language)
    
    try:
        os.remove(audio_path)
        print(f"Cleaned up: {audio_path}")
    except Exception as e:
        print(f"Error cleaning up: {e}")
    
    return f"Speech detected:\n{full_text}", sentences_map

# Ollama API URL
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"

intent_definitions = {
    "openness": "Use this intent when the sentence invites discussion, encourages collaboration, or emphasizes transparency or sharing. Example: 'Let’s share our ideas openly.'",
    "authority": "Use this intent when the sentence asserts control, commands action, or enforces a strict rule. Example: 'You must follow these instructions now.'",
    "leveler": "Use this intent when the sentence describes a clear progression from a lower to a higher state (e.g., small to big, weak to strong), but only when no numerical words like 'one', 'two', or 'three' are used unless 'first', 'second', or 'third' are present. If 'first', 'second', or 'third' is mentioned, always assign leveler, even if other intents apply. Do not use for vague stages without clear progression. Example: 'We moved from a small idea to a global movement.' or 'First, we plan; second, we execute.'",
    "listing": "Use this intent when the sentence explicitly states numbers less than 10 in word form (e.g., 'one', 'two', 'three', not '12' or 'twenty'). Example: 'We have three main goals.' or 'I’ll say it three times.'",
    "pointing": "Use this intent when the sentence directly addresses a specific individual. Example: 'John, this task is for you.'",
    "prompting": "Use this intent when the sentence encourages audience interaction or action, like telling them to do something. Example: 'Please raise your hand if you agree.'",
    "this&that": "Use this intent when the sentence contrasts two distinct options, ideas, or elements (e.g., different conditions or outcomes). Example: 'Some prefer coffee, but others choose tea.'",
    "description": "Use this intent when the sentence provides factual details, observations, or explanations without contrast, progression, or action. Example: 'The building is made of concrete.'",
    "emphasis": "Use this intent when the sentence highlights importance, urgency, or significance using strong language, but does not command action. Example: 'This opportunity is absolutely critical.'",
    "questioning": "Use this intent when the sentence poses a question, seeks clarification, or expresses curiosity, without necessarily demanding action. Example: 'What do you think about this idea?'"
}

sentence_classification_cache = {}
try:
    if os.path.exists("sentence_classification_cache.json"):
        with open("sentence_classification_cache.json", "r") as f:
            sentence_classification_cache = json.load(f)
        logger.info(f"Loaded {len(sentence_classification_cache)} entries from cache")
except Exception as e:
    logger.error(f"Error loading cache: {e}")
    sentence_classification_cache = {}

def classify_intent(sentence, sentences_map):
    if not sentence.strip():
        logger.warning("Empty sentence provided, returning None")
        return None

    if sentence in sentence_classification_cache:
        cached_result = sentence_classification_cache[sentence]
        if cached_result:
            time_info = find_best_matching_timestamp(sentence, sentences_map)
            cached_result["start"] = time_info.get("start", 0)
            cached_result["end"] = time_info.get("end", 0)
        logger.info(f"Cache hit for sentence: {sentence}")
        return cached_result
    
    prompt = f"""
You are an expert AI specializing in classifying sentences into one of ten intents: openness, authority, leveler, listing, pointing, prompting, this&that, description, emphasis, or questioning. Your task is to analyze a given sentence and assign the most appropriate intent based on strict criteria, ensuring high accuracy. Additionally, provide a confidence score (between 0 and 1) for the assigned intent, reflecting your certainty in the classification.

**Intent Definitions and Examples:**
- Openness: The sentence invites discussion, encourages collaboration, or emphasizes sharing. Example: "Let’s share our ideas openly."
- Authority: The sentence asserts control, commands action, or enforces a rule. Example: "You must follow these instructions now."
- Leveler: The sentence describes a clear progression from a lower to a higher state (e.g., small to big, weak to strong), but only when no numerical words like 'one', 'two', or 'three' are used unless 'first', 'second', or 'third' are present. If 'first', 'second', or 'third' is mentioned, always assign leveler, even if other intents apply. Do not use for vague stages without clear progression. Example: "We moved from a small idea to a global movement." or "First, we plan; second, we execute."
- Listing: The sentence explicitly includes numbers less than 10 in word form (e.g., 'one', 'two', 'three', not '12' or 'twenty'). Example: "We have three main goals." or "I’ll say it three times."
- Pointing: The sentence directly addresses a specific individual. Example: "John, this task is for you."
- Prompting: The sentence encourages audience interaction or action. Example: "Please raise your hand if you agree."
- This&that: The sentence contrasts two distinct options, ideas, or elements (e.g., different conditions or outcomes). Example: "Some prefer coffee, but others choose tea."
- Description: The sentence provides factual details, observations, or explanations without contrast, progression, or action. Example: "The building is made of concrete."
- Emphasis: The sentence highlights importance, urgency, or significance using strong language, but does not command action. Example: "This opportunity is absolutely critical."
- Questioning: The sentence poses a question, seeks clarification, or expresses curiosity, without necessarily demanding action. Example: "What do you think about this idea?"

**Task:**
- Analyze the sentence provided below.
- Assign exactly one intent that best matches the definitions and rules.
- Provide a confidence score (between 0 and 1) for the assigned intent.
- Return the output in **this exact format** (no extra words, no deviations):
  Intent: [intent] | Confidence: [confidence] | Sentence: [sentence]
- If no intent matches, return exactly:
  None

**Rules for Accuracy:**
1. Check for 'listing' first: If the sentence contains a number less than 10 in word form (e.g., 'one', 'two', 'three'), assign 'listing', unless 'first', 'second', or 'third' is present, which triggers 'leveler'.
2. Prioritize 'leveler' if 'first', 'second', or 'third' is present, ignoring other potential matches.
3. For 'this&that', confirm the sentence explicitly contrasts two elements (e.g., different places, conditions, or choices). Do not confuse with descriptions containing numbers.
4. For 'description', ensure the sentence is neutral and factual, without numbers in word form, contrast, command, or engagement.
5. For 'emphasis', verify the sentence uses strong language (e.g., 'critical', 'essential') but does not direct action like 'authority'.
6. For 'questioning', confirm the sentence is a question or seeks information, distinct from 'prompting' which expects action.
7. Avoid assigning intents based on weak matches; if uncertain after checking all rules, return None.
8. The confidence score should reflect the clarity of the match: high (0.9-1.0) for clear matches, medium (0.6-0.89) for reasonable but less certain matches, low (0.1-0.59) for weak matches.
9. Double-check that the output format is exact, preserving the input sentence verbatim.

**Sentence:** "{sentence}"
"""

    try:
        response = requests.post(OLLAMA_URL, json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False,
            "temperature": 0.0,
            "seed": 40
        })
        response.raise_for_status()
        result = response.json().get("response", "None").strip()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling Ollama API: {e}")
        return None

    if result.startswith("Intent:"):
        parts = result.split("|")
        if len(parts) == 3:
            intent_part = parts[0].replace("Intent:", "").strip().lower()
            confidence_part = parts[1].replace("Confidence:", "").strip()
            classified_sentence = parts[2].replace("Sentence:", "").strip()

            try:
                confidence = float(confidence_part)
                if not 0 <= confidence <= 1:
                    logger.warning(f"Invalid confidence score: {confidence_part}")
                    return None
            except ValueError:
                logger.warning(f"Invalid confidence format: {confidence_part}")
                return None

            if intent_part in intent_definitions:
                time_info = find_best_matching_timestamp(sentence, sentences_map)
                start_time = time_info.get("start", 0)
                end_time = time_info.get("end", 0)
                
                classification_result = {
                    "intent": intent_part.capitalize(),
                    "confidence": confidence,
                    "sentence": classified_sentence,
                    "start": start_time,
                    "end": end_time
                }
                
                sentence_classification_cache[sentence] = {
                    "intent": intent_part.capitalize(),
                    "confidence": confidence,
                    "sentence": classified_sentence
                }
                
                logger.info(f"Classified sentence: {sentence} as {intent_part} with confidence {confidence}")
                return classification_result

    sentence_classification_cache[sentence] = None
    logger.info(f"No intent matched for sentence: {sentence}")
    return None

def classify_paragraph(paragraph, sentences_map):
    if not paragraph.strip():
        logger.warning("Empty paragraph provided")
        return []
    
    paragraph = paragraph.replace(",", ", ").replace(".", ". ").replace("  ", " ")
    sentences = re.split(r'(?<=[.!?])\s+', paragraph.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    
    classified = []
    for sentence in sentences:
        result = classify_intent(sentence, sentences_map)
        if result:
            classified.append(result)
    
    paragraph_length = len(paragraph)
    return filter_results(classified, paragraph_length)

def filter_results(classified, paragraph_length):
    leveler_and_listing = [res for res in classified if res["intent"].lower() in ["leveler", "listing"]]
    other_intents = [res for res in classified if res["intent"].lower() not in ["leveler", "listing"]]

    if paragraph_length < 200:
        max_other_intents = 2
        max_leveler_and_listing = 100
    elif paragraph_length < 500:
        max_other_intents = 3
        max_leveler_and_listing = 100
    else:
        max_other_intents = 5
        max_leveler_and_listing = 100

    this_and_that = [res for res in other_intents if res["intent"].lower() == "this&that"]
    other_non_this_and_that = [res for res in other_intents if res["intent"].lower() != "this&that"]
    other_non_this_and_that = other_non_this_and_that[:max_other_intents - len(this_and_that)]
    
    return leveler_and_listing + this_and_that + other_non_this_and_that

# --- Process Video Function for Server ---
def process_video(video_path):
    if not os.path.exists(video_path):
        logger.error(f"Video file {video_path} does not exist.")
        return {"error": "Video file does not exist"}

    # Hand Movement Detection
    logger.info(f"Processing video for hand movements: {video_path}")
    hand_status, processing_time, hand_moved_times = process_video_for_hands(video_path)

    # Convert hand_moved_times to ranges
    hand_movement_ranges = []
    if hand_status and hand_moved_times:
        start = hand_moved_times[0]
        end = start
        for t in hand_moved_times[1:]:
            if t == end + 1:
                end = t
            else:
                hand_movement_ranges.append({"start": start, "end": end})
                start = end = t
        hand_movement_ranges.append({"start": start, "end": end})

    # Speech Transcription and Intent Classification
    logger.info(f"Processing video for speech and intent classification: {video_path}")
    result_text, sentences_map = transcribe_video_speech(video_path, language="en")
    
    paragraph = result_text.replace("Speech detected:\n", "")
    classified_sentences = classify_paragraph(paragraph, sentences_map)

    # Save cache
    try:
        with open("sentence_classification_cache.json", "w") as f:
            json.dump(sentence_classification_cache, f)
        logger.info(f"Saved {len(sentence_classification_cache)} entries to cache")
    except Exception as e:
        logger.error(f"Error saving cache: {e}")

    # Return structured output
    return {
        "handMovements": hand_movement_ranges,
        "intents": classified_sentences
    }

# --- Main Execution ---
def main():
    input_video = "e:\\GC\\test.mp4"
    result = process_video(input_video)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()