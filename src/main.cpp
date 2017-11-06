////////////////////////////////////////////////////////////////////////////////
//! Copyright 2017 Boitumelo Ruf. All rights reserved.
////////////////////////////////////////////////////////////////////////////////

// Std
#include <fstream>
#include <iostream>
#include <assert.h>

// Qt
#include <QApplication>
#include <QCoreApplication>
#include <QCommandLineParser>
#include <QCommandLineOption>
#include <QFile>
#include <QString>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "cudagaussianblur.cuh"

//==================================================================================================
void setupCommandlineOptions(QCommandLineParser& ioParser)
{
  // add help option
  ioParser.addHelpOption();

  // add positional argument for generation action
  ioParser.addPositionalArgument("image", "Input Image");

  // add option for output directory
  ioParser.addOption({{"o", "output"},
                     "Write output into file.", "file"});
}

//==================================================================================================
int main(int argc, char *argv[])
{
  //--- setup QCoreApplication ---
  QApplication app(argc, argv);
  QApplication::setApplicationName(TARGET_NAME);

  //--- setup command line parser ---
  QCommandLineParser cmdParser;
  setupCommandlineOptions(cmdParser);
  cmdParser.process(app);

  //--- get positional arguments of commandline parser ---
  QStringList positionalArgs = cmdParser.positionalArguments();
  if(positionalArgs.size() < 1) // if no argument is given print help
  {
    cmdParser.showHelp(1);
  }

  //--- check if input image exists ---
  if(!QFile(positionalArgs[0]).exists())
  {
    std::cout << "Input file does not exist!" << std::endl;
    return 1;
  }

  //--- read input image ---
  cv::Mat inputImgBGR = cv::imread(positionalArgs[0].toStdString());
  cv::Mat inputImgRGB;
  cv::cvtColor(inputImgBGR, inputImgRGB, CV_BGR2RGBA);
  cv::Mat outputImgRGB = runCudaGaussianBlur(inputImgRGB);
  cv::Mat outputImgBGR;
  cv::cvtColor(outputImgRGB, outputImgBGR, CV_RGBA2BGRA);

  //--- if output path is set write image into file ---
  if(cmdParser.isSet("o"))
  {
    QString outputPath = cmdParser.value("o");
    cv::imwrite(outputPath.toStdString(), outputImgBGR);
  }

  //--- display results ---
  const std::string inputWinName = "Input Image";
  const std::string outputWinName = "Output Image";
  cv::namedWindow(inputWinName);
  cv::namedWindow(outputWinName);
  cv::moveWindow(inputWinName, 100, 100);
  cv::moveWindow(outputWinName, 500, 100);
  cv::imshow(inputWinName, inputImgBGR);
  cv::imshow(outputWinName, outputImgBGR);
  cv::waitKey();

  return 0;
}
