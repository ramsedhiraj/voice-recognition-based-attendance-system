Voice-Based Attendance System using Machine Learning 🎙️
📘 Overview

The Voice-Based Attendance System is an intelligent solution that automates student or employee attendance marking using voice recognition.
The system leverages Machine Learning (ML) techniques for speaker identification, ensuring a contactless, secure, and efficient attendance process.

This project is part of the MCA (SY) curriculum under the guidance of Dr. V. C. Bagal, and developed by Dhiraj Ramse as a practical implementation of AI and ML in real-world systems.used for record the new recordings 

voice_attendance/
│
├── dataset/                         # Voice samples organized by student name
│   ├── Dhiraj/
│   │   ├── Dhiraj_1.wav
│   │   ├── Dhiraj_2.wav
│   └── NewStudent/
│       ├── NewStudent_1.wav
│
├── models/                          # Trained machine learning models
│   └── voice_model.pkl
│
├── attendance/                      # Auto-generated attendance records (CSV, XLSX)
│
├── gui.py                           # Main GUI application
├── train_model.py                   # Model training script
├── recognize_and_mark_attendance.py # Voice recognition & attendance marking
├── record_voice_samples.py          # Script to record samples for each student
├── requirements.txt                 # Dependency list
└── README.md                        # Project documentation
cmd to run the project

{
    python record_voice_samples.py
}

{
 train the model as well as update previous one 

"python train_model.py --finetune-path data --pretrained-out models/pretrained_model.pkl --model-out models/voice_model.pkl"

}

{
for "python recognize_and_mark_attendance.py"
}


Data set for training   
https://www.openslr.org/12/

future Scope 
1> show %
2> subject specefic attended lecture out of total lecture
3> total attendence in % 
4> subject specefic sttendence in % 
5> implement gui && design 

python libraries cmd
"""pip install numpy pandas librosa scikit-learn soundfile pyaudio wave xlsxwriter speechrecognition tkinter """

pip install numpy
pip install pandas
pip install librosa
pip install scikit-learn
pip install soundfile
pip install pyaudio
pip install wave
pip install xlsxwriter
pip install SpeechRecognition

Library Description
Library    	Purpose
numpy	    Mathematical operations & numerical data handling
pandas	    Create and manage CSV/XLSX attendance files
librosa   	Extract MFCC and spectral audio features
scikit-learn	Train the ML model (SVM classifier)
soundfile	Audio file reading/writing
pyaudio  	Real-time voice recording from mic
wave	    Save voice samples as .wav
xlsxwriter	Format Excel attendance reports
speechrecognition	Microphone input via GUI
tkinter	GUI framework for user interaction# voice-recognition-based-attendance-system
