import getpass


class AuthManager:
    def __init__(self):
        self.locked = False
        self.valid_username = "admin"
        self.valid_password = "airauth123"

    def toggle_lock(self):
        self.locked = not self.locked

    def is_locked(self):
        return self.locked

    def get_state_text(self):
        return "LOCKED" if self.locked else "UNLOCKED"

    def get_state_color(self):
        return (0, 0, 255) if self.locked else (0, 255, 0)

    def login(self, max_attempts=3):
        print("\n========== AirAuth-SOC Login ==========")
        print("Default username: admin")
        print("Default password: airauth123")
        print("Change these in src/auth_manager.py later.\n")

        for attempt in range(1, max_attempts + 1):
            username = input("Username: ").strip()
            password = getpass.getpass("Password: ")

            if username == self.valid_username and password == self.valid_password:
                print("\nLogin successful.\n")
                return True

            print(f"Invalid credentials. Attempt {attempt}/{max_attempts}\n")

        print("Too many failed attempts. Exiting.")
        return False
