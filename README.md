used for record the new recordings 
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
