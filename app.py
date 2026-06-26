import os
import subprocess
import sys

# Tenta instalar o mediapipe manualmente se não estiver instalado
try:
    import mediapipe
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mediapipe==0.10.9"])
    import mediapipe

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import av

st.title("SafeSignal AI - Protótipo")

class HandProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_hands = mediapipe.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.5)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resultados = self.hands.process(img_rgb)
        
        if resultados.multi_hand_landmarks:
            for hand_landmarks in resultados.multi_hand_landmarks:
                mediapipe.solutions.drawing_utils.draw_landmarks(img, hand_landmarks, mediapipe.solutions.hands.HAND_CONNECTIONS)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(key="example", video_processor_factory=HandProcessor)
