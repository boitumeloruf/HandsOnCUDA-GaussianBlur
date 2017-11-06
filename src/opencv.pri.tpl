OCV_VERS_STR = <edit-here>
GCC_VERS_STR = <edit-here>
OCV_ROOT_DIR = <edit-here>

# add directories and libraries to variables
unix {
	message("Using OpenCV "$${OCV_VERS_STR}", GCC "$${GCC_VERS_STR}")

	INCLUDEPATH += $${OCV_ROOT_DIR}/include/ \
		           $${OCV_ROOT_DIR}/include/opencv/
	DEPENDPATH += $${OCV_ROOT_DIR}/include/ \
		          $${OCV_ROOT_DIR}/include/opencv/

  CONFIG(debug, debug|release) {
    LIBS += -L$${OCV_ROOT_DIR}/lib/debug
    message("-- export LD_LIBRARY_PATH="$${OCV_ROOT_DIR}"/lib/debug:$LD_LIBRARY_PATH")
  }
  CONFIG(release, debug|release) {
    LIBS += -L$${OCV_ROOT_DIR}/lib/release
    message("-- export LD_LIBRARY_PATH="$${OCV_ROOT_DIR}"/lib/release:$LD_LIBRARY_PATH")
  }

  LIBS += -lopencv_core
  LIBS += -lopencv_highgui
  LIBS += -lopencv_imgcodecs
  LIBS += -lopencv_imgproc

}
