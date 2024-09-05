from ultralytics import YOLO

model = YOLO(r'C:\Users\msi1\Videos\Computer_Vision_Solution\Autonomous_Shoe_Spotter\Models_and_notebooks\best.pt')

results = model(source=0, show=True, conf=0.4, save=True)