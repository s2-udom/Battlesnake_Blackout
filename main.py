import os

# Import the fully initialized Flask app from our new optimized script
from rllib_agent import app

if __name__ == "__main__":
    # Gunicorn or Docker will pass the port via environment variables, defaulting to 8000
    port = int(os.environ.get("PORT", "8000"))
    print(f"Starting MAPPO Battlesnake Server on port {port}...")
    
    # Run the lightweight server
    app.run(host="0.0.0.0", port=port)