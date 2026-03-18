import torch
import cv2
import mediapipe as mp
from torchvision import transforms
from PIL import Image
from model import CNN


# ---------------------------
# 1. DEVICE
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------
# 2. LOAD MODEL
# ---------------------------
model = CNN(num_classes=26).to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()


# ---------------------------
# 3. TRANSFORMS
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])


# ---------------------------
# 4. CLASS LABELS
# ---------------------------
classes = [chr(i) for i in range(ord('A'), ord('Z') + 1)]


# ---------------------------
# 5. MEDIAPIPE HAND SETUP
# ---------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils


# ---------------------------
# 6. CAMERA
# ---------------------------
cap = cv2.VideoCapture(0)

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # convert to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # detect hands
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            # draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # get bounding box
            h, w, c = frame.shape
            x_list = []
            y_list = []

            for lm in hand_landmarks.landmark:
                x_list.append(int(lm.x * w))
                y_list.append(int(lm.y * h))

            x1, x2 = min(x_list), max(x_list)
            y1, y2 = min(y_list), max(y_list)

            # add padding
            padding = 20
            x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
            x2, y2 = min(w, x2 + padding), min(h, y2 + padding)

            # draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # crop hand
            hand_img = frame[y1:y2, x1:x2]

            try:
                # preprocess
                image = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)

                image = transform(image)
                image = image.unsqueeze(0).to(device)

                # predict
                with torch.no_grad():
                    output = model(image)
                    _, predicted = torch.max(output, 1)

                label = classes[predicted.item()]

                # display prediction
                cv2.putText(frame, f"{label}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2)

            except:
                pass

    cv2.imshow("Sign Language Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()