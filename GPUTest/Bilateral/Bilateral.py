import os
import unittest
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging

#
# Bilateral
#

class Bilateral(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Bilateral" # TODO make this more human readable by adding spaces
    self.parent.categories = ["Examples"]
    self.parent.dependencies = []
    self.parent.contributors = ["John Doe (AnyWare Corp.)"] # replace with "Firstname Lastname (Organization)"
    self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
It performs a simple thresholding on the input volume and optionally captures a screenshot.
"""
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc.
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""" # replace with organization, grant and thanks.

#
# BilateralWidget
#

class BilateralWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    # Instantiate and connect widgets ...

    #
    # Parameters Area
    #
    parametersCollapsibleButton = ctk.ctkCollapsibleButton()
    parametersCollapsibleButton.text = "Parameters"
    self.layout.addWidget(parametersCollapsibleButton)

    # Layout within the dummy collapsible button
    parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

    #
    # input volume selector
    #
    self.inputSelector = slicer.qMRMLNodeComboBox()
    self.inputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.inputSelector.selectNodeUponCreation = True
    self.inputSelector.addEnabled = False
    self.inputSelector.removeEnabled = False
    self.inputSelector.noneEnabled = False
    self.inputSelector.showHidden = False
    self.inputSelector.showChildNodeTypes = False
    self.inputSelector.setMRMLScene( slicer.mrmlScene )
    self.inputSelector.setToolTip( "Pick the input to the algorithm." )
    parametersFormLayout.addRow("Input Volume: ", self.inputSelector)

    #
    # output volume selector
    #
    self.outputSelector = slicer.qMRMLNodeComboBox()
    self.outputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.outputSelector.selectNodeUponCreation = True
    self.outputSelector.addEnabled = True
    self.outputSelector.removeEnabled = True
    self.outputSelector.noneEnabled = True
    self.outputSelector.showHidden = False
    self.outputSelector.showChildNodeTypes = False
    self.outputSelector.setMRMLScene( slicer.mrmlScene )
    self.outputSelector.setToolTip( "Pick the output to the algorithm." )
    parametersFormLayout.addRow("Output Volume: ", self.outputSelector)

    #
    # threshold value
    #

    self.kernelSizeSliderWidget = ctk.ctkSliderWidget()
    self.kernelSizeSliderWidget.singleStep = 1.0
    self.kernelSizeSliderWidget.minimum = 0.0
    self.kernelSizeSliderWidget.maximum = 20.0
    self.kernelSizeSliderWidget.value = 3.0
    self.kernelSizeSliderWidget.setToolTip("Set kernel size.")
    parametersFormLayout.addRow("Kernel Size:", self.kernelSizeSliderWidget)

    self.sigmaSpaceSliderWidget = ctk.ctkSliderWidget()
    self.sigmaSpaceSliderWidget.singleStep = 1.0
    self.sigmaSpaceSliderWidget.minimum = 0.1
    self.sigmaSpaceSliderWidget.maximum = 10.0
    self.sigmaSpaceSliderWidget.value = 5.0
    self.sigmaSpaceSliderWidget.setToolTip("Set sigma space.")
    parametersFormLayout.addRow("Sigma space:", self.sigmaSpaceSliderWidget)

    self.sigmaRangeSliderWidget = ctk.ctkSliderWidget()
    self.sigmaRangeSliderWidget.singleStep = 0.1
    self.sigmaRangeSliderWidget.minimum = 0.1
    self.sigmaRangeSliderWidget.maximum = 200.0
    self.sigmaRangeSliderWidget.value = 100.0
    self.sigmaRangeSliderWidget.setToolTip("Set sigma range.")
    parametersFormLayout.addRow("Sigma range:", self.sigmaRangeSliderWidget)

    #
    # Apply Button
    #
    self.applyButton = qt.QPushButton("Apply")
    self.applyButton.toolTip = "Run the algorithm."
    self.applyButton.enabled = False
    parametersFormLayout.addRow(self.applyButton)

    # connections
    self.applyButton.connect('clicked(bool)', self.onApplyButton)
    self.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.outputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)

    # Add vertical spacer
    self.layout.addStretch(1)

    # Refresh Apply button state
    self.onSelect()

  def cleanup(self):
    pass

  def onSelect(self):
    self.applyButton.enabled = self.inputSelector.currentNode() and self.outputSelector.currentNode()

  def onApplyButton(self):
    logic = BilateralLogic()
    logic.run(self.inputSelector.currentNode(), self.outputSelector.currentNode(), self.kernelSizeSliderWidget.value, self.sigmaSpaceSliderWidget.value, self.sigmaRangeSliderWidget.value/32767.0)

#
# BilateralLogic
#

class BilateralLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def hasImageData(self,volumeNode):
    """This is an example logic method that
    returns true if the passed in volume
    node has valid image data
    """
    if not volumeNode:
      logging.debug('hasImageData failed: no volume node')
      return False
    if volumeNode.GetImageData() is None:
      logging.debug('hasImageData failed: no image data in volume node')
      return False
    return True

  def isValidInputOutputData(self, inputVolumeNode, outputVolumeNode):
    """Validates if the output is not the same as input
    """
    if not inputVolumeNode:
      logging.debug('isValidInputOutputData failed: no input volume node defined')
      return False
    if not outputVolumeNode:
      logging.debug('isValidInputOutputData failed: no output volume node defined')
      return False
    if inputVolumeNode.GetID()==outputVolumeNode.GetID():
      logging.debug('isValidInputOutputData failed: input and output volume is the same. Create a new volume for output to avoid this error.')
      return False
    return True

  def getFragmentShaderCode(self):
    return """
//VTK::System::Dec
varying vec2 tcoordVSOutput;
uniform float zPos;
//VTK::AlgTexUniforms::Dec
//VTK::Output::Dec
 uniform vec3 pixelToTexture0;
 uniform int kernelSize;
 uniform float sigmaSpace;
 uniform float sigmaRange;
 const float pi = 3.1415926535897932384626433832795;
 const float sqrt_2_pi = 2.5066282746310002;
 float gaussian ( float v, float sigma, float sigma_squared ) {
   float num = exp ( - ( v*v ) / ( 2.0 * sigma_squared ));
   float den = sqrt_2_pi * sigma;
   return num / den;
 }
 // From https://people.csail.mit.edu/sparis/bf_course/course_notes.pdf
 void doBilateralFilter()
 {
   vec3 interpolatedTextureCoordinate = vec3(tcoordVSOutput, zPos);
   float sigmaSpaceSquared = sigmaSpace * sigmaSpace;
   float sigmaRangeSquared = sigmaRange * sigmaRange;
   float background = float(texture(inputTex0, interpolatedTextureCoordinate).r);
   float v = 0.0;
   float w = 0.0;
   for (int i = -kernelSize; i <= kernelSize; i++) {
     for (int j = -kernelSize; j <= kernelSize; j++) {
       for (int k = -kernelSize; k <= kernelSize; k++) {
         vec3 offset = vec3(i,j,k) * pixelToTexture0;
         vec3 neighbor = interpolatedTextureCoordinate + offset;
         float neighborStrength = float(texture(inputTex0, neighbor).r);
         float ww = 0.0;
         ww = gaussian ( distance(offset, interpolatedTextureCoordinate), sigmaSpace, sigmaSpaceSquared ) ;
         ww *= gaussian ( background - neighborStrength, sigmaRange, sigmaRangeSquared );
         w += ww;
         v += ww * neighborStrength;
       }
     }
   }
   v = v / w;
   gl_FragData[0] = vec4(vec3(v),1.0); // cast if needed
 }
 void main()
 {
   doBilateralFilter();
 }
    """

  def run(self, inputVolume, outputVolume, kernelSize, sigmaSpace, sigmaRange):
    """
    Run the actual algorithm
    """

    if not self.isValidInputOutputData(inputVolume, outputVolume):
      slicer.util.errorDisplay('Input volume is the same as output volume. Choose a different output volume.')
      return False

    logging.info('Processing started')

    imageToGPUFilter = vtk.vtkImageToGPUImageFilter()
    imageToGPUFilter.SetInputDataObject(inputVolume.GetImageData())
    bilateralGPUFilter = vtk.vtkGPUSimpleImageFilter()
    bilateralGPUFilter.SetInputConnection(imageToGPUFilter.GetOutputPort())
    bilateralGPUFilter.GetShaderProperty().SetFragmentShaderCode(self.getFragmentShaderCode())
    pixelToTexture0 = [0,0,0]
    inputVolume.GetImageData().GetDimensions(pixelToTexture0)
    for i in range(3):
      pixelToTexture0[i] = 1/pixelToTexture0[i]
    print("pixelToTexture0: " + str(pixelToTexture0[0]) + ", " + str(pixelToTexture0[1]) + ", " + str(pixelToTexture0[2]))
    print("kernelSize: " + str(kernelSize))
    print("sigmaSpace: " + str(sigmaSpace))
    print("sigmaRange: " + str(sigmaRange))
    bilateralGPUFilter.GetShaderProperty().GetFragmentCustomUniforms().SetUniform3f("pixelToTexture0", pixelToTexture0)
    bilateralGPUFilter.GetShaderProperty().GetFragmentCustomUniforms().SetUniformi("kernelSize", int(kernelSize))
    bilateralGPUFilter.GetShaderProperty().GetFragmentCustomUniforms().SetUniformf("sigmaSpace", sigmaSpace)
    bilateralGPUFilter.GetShaderProperty().GetFragmentCustomUniforms().SetUniformf("sigmaRange", sigmaRange)
    bilateralGPUFilter.SetOutputScalarTypeToShort()
    gpuToImageFilter = vtk.vtkGPUImageToImageFilter()
    gpuToImageFilter.SetInputConnection(bilateralGPUFilter.GetOutputPort())
    gpuToImageFilter.Update()

    outputVolume.SetAndObserveImageData(gpuToImageFilter.GetOutput())
    matrix = vtk.vtkMatrix4x4()
    inputVolume.GetIJKToRASMatrix(matrix)
    outputVolume.SetIJKToRASMatrix(matrix)

    logging.info('Processing completed')

    return True


class BilateralTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear(0)

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_Bilateral1()

  def test_Bilateral1(self):
    """ Ideally you should have several levels of tests.  At the lowest level
    tests should exercise the functionality of the logic with different inputs
    (both valid and invalid).  At higher levels your tests should emulate the
    way the user would interact with your code and confirm that it still works
    the way you intended.
    One of the most important features of the tests is that it should alert other
    developers when their changes will have an impact on the behavior of your
    module.  For example, if a developer removes a feature that you depend on,
    your test should break so they know that the feature is needed.
    """

    self.delayDisplay("Starting the test")
    #
    # first, get some data
    #
    import SampleData
    SampleData.downloadFromURL(
      nodeNames='FA',
      fileNames='FA.nrrd',
      uris='http://slicer.kitware.com/midas3/download?items=5767',
      checksums='SHA256:12d17fba4f2e1f1a843f0757366f28c3f3e1a8bb38836f0de2a32bb1cd476560')
    self.delayDisplay('Finished with download and loading')

    volumeNode = slicer.util.getNode(pattern="FA")
    logic = BilateralLogic()
    self.assertIsNotNone( logic.hasImageData(volumeNode) )
    self.delayDisplay('Test passed!')
