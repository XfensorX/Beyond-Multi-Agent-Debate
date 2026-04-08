import json


class ExceptionEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Exception):
            # Capture the traceback for the current exception instance if available
            # Note: A simple obj does not carry full traceback info without sys.exc_info()
            return {
                "type": obj.__class__.__name__,
                "module": obj.__class__.__module__,
                "message": str(obj),
                "args": getattr(obj, "args", ()),
                # Traceback can only be accurately captured at the point of exception handling with sys.exc_info() or traceback.format_exc()
            }
        # Let the base class handle other objects
        return json.JSONEncoder.default(self, obj)
