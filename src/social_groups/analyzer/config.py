from __future__ import annotations

MAX_PARALLEL_REQUESTS = 50
MAX_IDS_PER_REQUEST = 10
IN_QUEUE_MAXSIZE = 10000
OUT_QUEUE_MAXSIZE = 10000
CHUNK_SIZE = 500
MAX_RETRIES = 2**30
QUEUE_TIMEOUT = 10  # seconds

PARALLEL_FILE_WRITES = 30


# maps know faulty span_Ids to the data that should be returned
ATTRIBUTE_KEY_SPAN_URL = "custom_phoenix_span_url_attribute"
KNOWN_FAULTY_SPAN_IDS = {
    # Only Used in old data
    # "b150eaf387158101": {
    #     "question": '{"question_id":9689,"question":"At 303 . K, the vapor pressure of benzene is 120 . Torr and that of hexane is 189 Torr. Calculate the vapor pressure of a solution for which $x_{\\\\text {benzene }}=0.28$ assuming ideal behavior.","src":"scibench-thermo","category":"physics","cot_content":"","answer_index":0,"answer":"A","options":["170 $\\\\mathrm{Torr}$","210 $\\\\mathrm{Torr}$","130 Torr","200 Torr","190 $\\\\mathrm{Torr}$","140 Torr","220 Torr","180 Torr","160 Torr","150 $\\\\mathrm{Torr}$"]}',
    #     ATTRIBUTE_KEY_SPAN_URL: "unknown",
    # }
}
