"""GazeDot launcher with a console (for troubleshooting)."""

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    try:
        from eyetrack.app import main

        main()
    except Exception:
        traceback.print_exc()
        input("\nエラーが発生しました。Enter で閉じます...")
