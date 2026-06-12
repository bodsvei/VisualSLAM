import g2o

def check():
    try:
        # These are the specific types your project requires
        edge = g2o.EdgeSE3ProjectXYZ()
        print("✅ SUCCESS: g2o installed with EdgeSE3ProjectXYZ support.")
    except AttributeError:
        print("⚠️ WARNING: g2o installed, but EdgeSE3ProjectXYZ is missing.")
    except Exception as e:
        print(f"❌ ERROR: g2o not found or failed to load. Details: {e}")

if __name__ == "__main__":
    check()