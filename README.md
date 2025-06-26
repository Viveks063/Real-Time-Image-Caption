# 🖼Real-Time Image Captioning with Voice Output 🎙️

A Python-based AI application that uses your webcam to generate **real-time captions** for the visual scene using **Salesforce's BLIP model**, and then **reads the caption aloud** using `pyttsx3` text-to-speech engine. This combines **Computer Vision**, **NLP**, and **Speech Synthesis** into one powerful, fun tool.

## Features

- Captures live video from webcam
- Uses **BLIP** (Bootstrapping Language-Image Pretraining) model for image captioning
- Converts generated captions into voice using `pyttsx3`
- Displays real-time captions on the video feed
- Runs on **CPU or GPU** automatically

---

## Tech Stack

- Python
- OpenCV (video capture + display)
- Hugging Face Transformers (BLIP model)
- Torch (PyTorch backend)
- Pillow (PIL)
- pyttsx3 (offline TTS)

