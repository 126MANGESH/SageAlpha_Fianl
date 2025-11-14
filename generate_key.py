import secrets
key = secrets.token_urlsafe(50)  # Generates a 50-character URL-safe random string
print(f"FLASK_SECRET_KEY={key}")
