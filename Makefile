TARGETS=openpose_image_coco openpose_video_coco
CXX=g++
CXXFLAGS=`pkg-config --cflags --libs opencv4`

all: $(TARGETS)

openpose_image_coco: OpenPoseImageCoco.cpp
	$(CXX) OpenPoseImageCoco.cpp -o $@ $(CXXFLAGS)

openpose_video_coco: OpenPoseVideo.cpp
	$(CXX) OpenPoseVideo.cpp -o $@ $(CXXFLAGS)

clean:
	rm -f $(TARGETS)
