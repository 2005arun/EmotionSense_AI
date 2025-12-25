"""
EmotionSense - Main Entry Point
Run the application from the project root
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit application"""
    # Get the directory of this script
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the Streamlit app
    app_path = os.path.join(project_root, 'frontend', 'app.py')
    
    # Run Streamlit
    subprocess.run([
        sys.executable, '-m', 'streamlit', 'run',
        app_path,
        '--server.headless', 'true',
        '--browser.gatherUsageStats', 'false'
    ])


if __name__ == '__main__':
    main()
