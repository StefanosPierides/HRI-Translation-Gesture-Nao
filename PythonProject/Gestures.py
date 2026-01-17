import cv2
import mediapipe as mp
import socket
import os
import time
from threading import Thread, Lock

# ========================================================================
#   SERVER CONFIGURATION
# ========================================================================
# '0.0.0.0' allows connections from any device on the network (PC + Robot)
host = '0.0.0.0'
port = 8888


class SimpleServer:
    """
    Acts as the 'Central Hub' or 'Router' for the entire system.

    Responsibilities:
    1. Accepts connections from:
       - The Translation Controller (Laptop script)
       - The NAO Robot (Choregraphe script)
    2. Broadcasts data:
       - Sends live 'Gesture' strings from the camera to the Robot.
       - Relays 'Speech' commands (SAY:...) from Laptop -> Robot.
       - Relays 'Replay' commands (FORCE:...) from Laptop -> Robot.
    3. Handles Priority:
       - Uses 'override_until' to pause live camera data when a recorded
         gesture sequence is being replayed, ensuring smooth playback.
    """

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.server_socket = None
        self.clients = []  # List of all connected sockets (Robot + Laptop)
        self.lock = Lock()  # Thread-safe lock for accessing the client list
        self.override_until = 0  # Timestamp: Ignore camera input until this time
        self.start_server()

    def start_server(self):
        """Initializes the TCP server and starts the listener thread."""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # REUSEADDR allows restarting the script without 'Address already in use' errors
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)  # Allow up to 5 simultaneous connections

            # Start accepting connections in the background so main loop can run camera
            Thread(target=self.accept_connections, daemon=True).start()
            print(f"Server listening on {self.host}:{self.port}")
        except Exception as e:
            print(f"Failed to start server: {e}")

    def accept_connections(self):
        """Constantly waits for new devices (Robot/PC) to connect."""
        while True:
            try:
                client_socket, client_address = self.server_socket.accept()
                print(f"New Connection from {client_address}")

                with self.lock:
                    self.clients.append(client_socket)

                # Spawn a dedicated listener thread for EACH client.
                # This allows us to receive messages (like 'SAY:') from the Laptop.
                Thread(target=self.handle_client, args=(client_socket,), daemon=True).start()
            except Exception as e:
                print(f"Connection acceptance error: {e}")

    def handle_client(self, client_socket):
        """
        Listens for incoming messages FROM a specific client.
        Mainly used to catch commands from 'TranslationController.py'.
        """
        while True:
            try:
                data = client_socket.recv(1024)
                if not data: break  # Client disconnected
                message = data.decode('utf-8')

                # --- ROUTING LOGIC ---

                # Case 1: Speech Command (Laptop -> Robot)
                # Format: "SAY:Hello World"
                if "SAY:" in message:
                    print(f"Relaying Speech: {message}")
                    self.send_signal(message)

                # Case 2: Replay Command (Laptop -> Robot)
                # Format: "FORCE:Right_Open_Palm"
                # This overrides the live camera for a split second to ensure
                # the robot mimics the recorded gesture, not the current stillness.
                elif "FORCE:" in message:
                    gesture = message.split("FORCE:")[1]
                    # Send immediately
                    self.send_signal(gesture)
                    # Set a timeout: Ignore camera for 0.3s
                    self.override_until = time.time() + 0.3

            except:
                break

        # Cleanup on disconnect
        with self.lock:
            if client_socket in self.clients:
                self.clients.remove(client_socket)
            print("Client disconnected.")

    def send_signal(self, message):
        """
        Broadcasts a message to ALL connected clients.
        Used for:
        - Sending detected gestures to Robot.
        - Relaying speech commands to Robot.
        """
        with self.lock:
            for client in self.clients:
                try:
                    client.sendall(message.encode())
                except:
                    # If sending fails, assume client is dead; it will be removed by handle_client
                    pass


def detect_gestures(server_host, server_port):
    """
    Main Execution Loop:
    1. Starts the Server.
    2. Opens the Webcam.
    3. Runs MediaPipe Hand Tracking.
    4. Sends recognized gestures to the Server (which broadcasts to Robot).
    """
    print("Starting server...")
    server = SimpleServer(server_host, server_port)

    # Open Default Camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera error: Could not open webcam.")
        return

    # --- MediaPipe Setup ---
    mp_hands = mp.tasks.vision
    BaseOptions = mp.tasks.BaseOptions
    GestureRecognizer = mp_hands.GestureRecognizer
    GestureRecognizerOptions = mp_hands.GestureRecognizerOptions
    VisionRunningMode = mp_hands.RunningMode

    # --- Dynamic Path Finding ---
    # Locates the 'models' folder relative to this script script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "models", "gesture_recognizer.task")

    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return

    # Configure Recognizer
    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,  # Optimized for streaming
        num_hands=2  # Track both hands
    )

    recognizer = GestureRecognizer.create_from_options(options)
    last_output = None

    print("Gesture recognition started...")

    while True:
        ret, frame = cap.read()
        if not ret: continue

        # Resize & Convert for MediaPipe
        frame = cv2.resize(frame, (480, 480))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Run Inference
        timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        result = recognizer.recognize_for_video(mp_image, timestamp)

        output_list = []

        # Parse Results (Left/Right Hand + Gesture Name)
        if result.gestures:
            for i, gesture_list in enumerate(result.gestures):
                gesture_name = gesture_list[0].category_name

                # Identify Handedness (Left/Right)
                if result.handedness and len(result.handedness) > i:
                    handed = result.handedness[i][0].category_name
                else:
                    handed = "Unknown"

                output_list.append(f"{handed}_{gesture_name}")

        if not output_list:
            output_list.append("NONE")

        final_output = " | ".join(output_list)

        # --- LOGIC UPDATE: OVERRIDE CHECK ---
        # If 'override_until' is active (meaning a REPLAY is happening),
        # we SKIP sending the live camera data. This prevents the live video
        # from fighting with the recorded replay data.
        if time.time() > server.override_until:
            if final_output != last_output:
                print("Gesture:", final_output)
                server.send_signal(final_output)
                last_output = final_output
        # ------------------------------------

        # Draw UI
        cv2.putText(frame, final_output, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow("Gesture Recognition", frame)

        # Exit on 'ESC'
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_gestures(host, port)