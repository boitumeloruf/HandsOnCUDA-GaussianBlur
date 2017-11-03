# set output directories
unix {
  # build directory
  BUILD_DIR = $$PWD

  # set dest dir dependent on template
  equals(TEMPLATE,"lib") {
    DESTDIR=$$BUILD_DIR/lib
  }
  equals(TEMPLATE, "app") {
    DESTDIR=$$BUILD_DIR/bin
  }

  # set temp directory
  AUX_DIR = $$BUILD_DIR/aux/$$MODULENAME
  OBJECTS_DIR = $$AUX_DIR/obj
  MOC_DIR = $$AUX_DIR/moc
  RCC_DIR = $$AUX_DIR/rcc
  UI_DIR = $$AUX_DIR/ui
}


# add suffix "_debug" to debug output
CONFIG(debug, debug|release) {
  unix: TARGET = $$join(TARGET,,,_debug)
}
