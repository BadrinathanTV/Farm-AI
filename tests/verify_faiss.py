import faiss
import sys

print(f"Faiss version: {faiss.__version__}")
try:
    # This might fail if no driver, but let's check
    # If the module has these attributes, it's the GPU version
    if hasattr(faiss, 'StandardGpuResources'):
        print("StandardGpuResources found in module (GPU build confirmed)")
        try:
             res = faiss.StandardGpuResources()
             print("StandardGpuResources instantiated successfully")
        except Exception as e:
             print(f"Could not instantiate GPU resources (Driver issue?): {e}")
    else:
        print("StandardGpuResources NOT found (Is this the CPU build?)")

except Exception as e:
    print(f"Error checking GPU: {e}")

# Verify basic functionality (CPU fallback valid?)
try:
    index = faiss.IndexFlatL2(10)
    print("IndexFlatL2 (CPU) created successfully")
except Exception as e:
    print(f"Failed to create basic index: {e}")
