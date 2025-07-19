PYTHON=python 
# SCRIPT=dual_detect_sort.py
SCRIPT=final_program_2.py


RACK=rack.pt
# OBJ=best_obj.pt 
OBJ = yolo11s_latest.pt
SOURCE=test.mov
THRESH=0.5
RES=1280x720

# Set RECORD=1 when calling make to enable video recording
ifeq ($(RECORD),1)
	RECORD_FLAG=--record
else
	RECORD_FLAG=
endif

run:
	$(PYTHON) $(SCRIPT) --rack_model $(RACK) --obj_model $(OBJ) --source $(SOURCE) --thresh $(THRESH) $(RECORD_FLAG) --resolution $(RES)

