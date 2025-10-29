import tkinter as tk
from tkinter import messagebox
import subprocess
import os

# ----------------------------
# Folder setup check
# ----------------------------
required_folders = ["data", "models", "attendance"]
for folder in required_folders:
    os.makedirs(folder, exist_ok=True)

# ----------------------------
# Command helper
# ----------------------------
def run_script(script_name):
    """Run another Python script from the same folder."""
    try:
        subprocess.run(["python", script_name], check=True)
    except subprocess.CalledProcessError:
        messagebox.showerror("Error", f"❌ Failed to run {script_name}")
    except FileNotFoundError:
        messagebox.showerror("Error", f"⚠️ {script_name} not found in directory!")


# ----------------------------
# Button actions
# ----------------------------
def record_samples():
    run_script("record_voice_samples.py")

def train_model():
    run_script("train_model.py")

def mark_attendance():
    run_script("recognize_and_mark_attendance.py")

def view_attendance():
    if not os.path.exists("attendance/"):
        messagebox.showerror("Error", "No attendance folder found!")
        return
    csv_files = [f for f in os.listdir("attendance/") if f.endswith(".csv")]
    if not csv_files:
        messagebox.showinfo("Attendance", "No attendance records found yet.")
        return
    last_file = os.path.join("attendance", sorted(csv_files)[-1])
    os.system(f'start excel "{last_file}"')


# ----------------------------
# GUI Setup
# ----------------------------
root = tk.Tk()
root.title("🎙 Voice Attendance System")
root.geometry("500x500")
root.config(bg="#eef3f7")

tk.Label(root, text="Voice-Based Attendance System", font=("Arial", 16, "bold"), bg="#eef3f7").pack(pady=20)

tk.Button(root, text="🎤 Record Student Samples", font=("Arial", 12), width=30, command=record_samples).pack(pady=10)
tk.Button(root, text="🧠 Train Voice Model", font=("Arial", 12), width=30, command=train_model).pack(pady=10)
tk.Button(root, text="✅ Mark Attendance", font=("Arial", 12), width=30, command=mark_attendance).pack(pady=10)
tk.Button(root, text="📄 View Attendance Sheet", font=("Arial", 12), width=30, command=view_attendance).pack(pady=10)
tk.Button(root, text="❌ Exit", font=("Arial", 12), width=30, command=root.quit).pack(pady=20)

tk.Label(root, text="Developed by Dhiraj | MCA Project", bg="#eef3f7", fg="gray", font=("Arial", 9)).pack(side="bottom", pady=10)

root.mainloop()
