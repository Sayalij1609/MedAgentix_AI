import os
from app import create_app

# Instantiate the Flask app using the Application Factory
app = create_app()

if __name__ == '__main__':
    # Retrieve port from environment, fallback to clinical standard 5000
    port = int(os.environ.get('PORT', 5000))
    
    print(f" * MedAgentix AI Backend instantiating...")
    print(f" * Synced allowed CORS origins: http://localhost:5173")
    
    # Run development WSGI server
    app.run(host='0.0.0.0', port=port, debug=True)
