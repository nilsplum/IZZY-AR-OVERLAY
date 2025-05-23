// Global function to log errors
function logError(methodName, error) {
  console.error(`Error in ${methodName}:`, error);

  // Get or create the ErrorLog div
  let errorDiv = document.getElementById('ErrorLog');
    errorDiv.id = 'ErrorLog';

  // Append the error message and stack trace
  errorDiv.innerHTML += `<p>Error in ${methodName}: ${error.message}<br>${error.stack}<br></p>`;
}

class FrameData {
  constructor(width, height) {
    try {
      // Initialize reusable Mats for downscaled processing resolution
      this.srcMat = new cv.Mat(height, width, cv.CV_8UC4);
      this.grayMat = new cv.Mat(height, width, cv.CV_8UC1);
      this.keypoints = new cv.KeyPointVector();
      this.descriptors = new cv.Mat();
      this.goodMatches = new cv.DMatchVector();
      this.transformationMatrix = null;
      this.qualityIndicator = null;
    } catch (error) {
      logError('FrameData.constructor', error);
    }
  }

  update(imageData) {
    try {
      // Update srcMat with new image data
      this.srcMat.data.set(imageData.data);

      // Convert to grayscale for feature detection
      cv.cvtColor(this.srcMat, this.grayMat, cv.COLOR_BGRA2GRAY);

      // Clear and reinitialize keypoints and descriptors
      this.keypoints = new cv.KeyPointVector();
      this.descriptors.delete();
      this.descriptors = new cv.Mat();
      this.goodMatches = new cv.DMatchVector(); // Clear matches
    } catch (error) {
      logError('FrameData.update', error);
    }
  }

  setQualityIndicator(qualityValue) {
    try {
      this.qualityIndicator = qualityValue;
    } catch (error) {
      logError('FrameData.setQualityIndicator', error);
    }
  }

  // Clean up Mats when no longer needed
  delete() {
    try {
      this.srcMat.delete();
      this.grayMat.delete();
      this.keypoints.delete();
      this.descriptors.delete();
      this.goodMatches.delete();
    } catch (error) {
      logError('FrameData.delete', error);
    }
  }
}

class ARFeatureMatcher {
  constructor(canvas, referenceImageUrl, overlayImagePath) {
    try {
      // The higher, the more features are detected which leads to more matches but also more false positives
      this.featureDetectionSensitivity = 0.0005;
      // The larger, the more accurate the matches but also the more time needs the method "this.featureDetector.detectAndCompute()" which is the most time consuming method
      this.initialProcessingCanvasWidth = 100;

      // PERFORMANCE AND QUALITY SETTINGS
      // Configurable canvas size for processing
      // Indirectly how many features are detected
      // Features are possible matches
      this.matchDistanceThreshold = 0.75; // Lowe's ratio test threshold for cross-checking
      // Initialize essential properties

      // For blocking the overlay when quality is poor
      this.displayingThresholdQuality = 0.06; // Minimum quality indicator for displaying overlay

      // Range over which the opacity fades from fully opaque to fully transparent
      this.qualityIndicatorFadeRange = 0.015; // Adjust as needed
      // Calculate the maximum quality indicator value where opacity becomes zero
      this.qualityIndicatorMax = this.displayingThresholdQuality + this.qualityIndicatorFadeRange;

      // Frame rate and processing canvas size adjustment variables
      this.targetFrameRate = 10; // Target frame rate in fps
      this.minFrameRate = 7; // Minimum frame rate

      this.minProcessingCanvasWidth = 250; // Minimum processing canvas width
      this.processingCanvasWidthStep = 20; // Amount to increase/decrease width per adjustment

      this.nFramesForAveraging = 6; // This can be adjusted as needed
      this.qualityHistory = []; // To store recent quality indicators
      this.averageQualityIndicator = 0; // The rolling average of quality indicators

      this.desiredZoomFactor = 2; // Desired zoom factor for the camera

      // Canvas to finally display the video feed and overlay
      this.displayingCanvas = canvas;
      this.displayingContext = canvas.getContext("2d", {
        willReadFrequently: true,
      });

      // Off-screen canvas for lower resolution processing
      this.processingCanvas = document.createElement("canvas");
      this.processingContext = this.processingCanvas.getContext("2d", {
        willReadFrequently: true,
      });

      // Off-screen canvas for holding the transformed overlay image
      this.transformedCanvas = document.createElement("canvas");
      this.transformedContext = this.transformedCanvas.getContext("2d", {
        willReadFrequently: true,
      });

      this.referenceImageUrl = referenceImageUrl;
      this.overlayImagePath = overlayImagePath;

      // PNG Set animation images and current frame index
      this.overlayPNGs = [];
      this.currentFrameIndex = 0;

      // Initialization
      this.isOpenCVInitialized = false;
      this.processing = false;

      // Current frame data
      this.frameData = null;

      this.lastFrameTime = Date.now(); // Initialize last frame time

      this.currentProcessingCanvasWidth = this.initialProcessingCanvasWidth; // Initial processing canvas width
      this.aspectRatio = null; // Will be set after video metadata is loaded
    } catch (error) {
      logError('ARFeatureMatcher.constructor', error);
    }
  }

  async initialize() {
    try {
      // Initialize OpenCV and camera, load reference images
      await this.initializeOpenCV();
      await this.setupCVDependentProperties();
      await this.initializeCamera();
      await this.loadReferenceImage(this.referenceImageUrl);
      await this.loadOverlayPNGs();
      // Wait for the video to be ready before starting processing
      await this.waitForVideoReady();
      this.startProcessing();
    } catch (err) {
      // Existing error handling
      console.error("Initialization error: ", err);
      // Log the error
      logError('initialize', err);
    }
  }

  // Asynchronously initialize OpenCV.js
  initializeOpenCV() {
    try {
      return new Promise((resolve, reject) => {
        try {
          if (typeof cv === "undefined") {
            alert("OpenCV.js not loaded");
            reject("OpenCV.js not loaded");
            return;
          }

          cv["onRuntimeInitialized"] = () => {
            this.isOpenCVInitialized = true;
            resolve();
          };

          if (cv.Mat) {
            this.isOpenCVInitialized = true;
            resolve();
          }
        } catch (err) {
          console.error("OpenCV initialization error: ", err);
          reject(err);
        }
      });
    } catch (error) {
      logError('initializeOpenCV', error);
    }
  }

  waitForVideoReady() {
    try {
      return new Promise((resolve) => {
        if (this.video.readyState >= 2) {
          // Video is ready
          resolve();
        } else {
          // Wait for the video to become ready
          this.video.onloadeddata = () => {
            resolve();
          };
        }
      });
    } catch (error) {
      logError('waitForVideoReady', error);
    }
  }

  // Set up OpenCV-dependent properties like feature detector and matcher
  async setupCVDependentProperties() {
    try {
      if (this.isOpenCVInitialized) {
        this.featureDetector = new cv.AKAZE();
        this.featureDetector.setThreshold(this.featureDetectionSensitivity); // Adjust detection sensitivity
        this.matcher = new cv.BFMatcher(); // Matcher for feature matching
      } else {
        console.error("OpenCV not initialized.");
      }
    } catch (error) {
      logError('setupCVDependentProperties', error);
    }
  }

  // Initialize the camera stream and adjust canvas dimensions based on the video feed
  async initializeCamera() {
    try {
      // Check if camera APIs are supported
      await this.checkCameraSupport();

      // Get the camera stream
      const stream = await this.getCameraStream();

      // Set up the video element with the obtained stream
      await this.setupVideoElement(stream);

      // Apply zoom if supported
      await this.applyZoomIfSupported(stream);

      // Start video playback
      await this.startVideoPlayback();
    } catch (err) {
      console.error("Camera initialization error: ", err);

      // Handle errors and execute default behavior
      await this.handleCameraError(err);
      // Log the error
      logError('initializeCamera', err);
    }
  }

  // Check if navigator.mediaDevices and getUserMedia are supported
  async checkCameraSupport() {
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        this.displayGetCloserMessage(
          "Camera API is not supported on this device or browser."
        );
        throw new Error("Camera API not supported");
      }

      // Check the permission status for the camera
      try {
        const permissionStatus = await navigator.permissions.query({
          name: "camera",
        });
        if (permissionStatus.state === "denied") {
          this.displayGetCloserMessage(
            "Camera access has been denied. Please enable it in your browser settings."
          );
          throw new Error("Camera access denied");
        }
      } catch (err) {
        // Some browsers may not support navigator.permissions
        console.warn(
          "Permissions API not supported, proceeding without checking permissions."
        );
      }
    } catch (error) {
      logError('checkCameraSupport', error);
      throw error; // Re-throw to be caught in initializeCamera
    }
  }

  // Get the camera stream with desired constraints
  async getCameraStream() {
    try {
      const constraints = {
        video: {
          facingMode: { ideal: "environment" }, // Attempt to use back camera
          // You can add more constraints here if needed
        },
      };
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      return stream;
    } catch (err) {
      // Rethrow the error to be handled in initializeCamera
      logError('getCameraStream', err);
      throw err;
    }
  }

  // Set up the video element with the obtained stream
  setupVideoElement(stream) {
    try {
      return new Promise((resolve) => {
        // Create video element
        this.video = document.createElement("video");
        this.video.setAttribute("playsinline", "true");
        this.video.muted = true;
        this.video.style.display = "none"; // Hide the video element from view
        document.body.appendChild(this.video);

        // Set the video stream as the source for the video element
        this.video.srcObject = stream;

        // Wait for the video metadata to load (dimensions, etc.)
        this.video.onloadedmetadata = () => {
          this.adjustCanvasSizes();
          resolve();
        };
      });
    } catch (error) {
      logError('setupVideoElement', error);
    }
  }

  // Adjust canvas sizes based on video dimensions
  adjustCanvasSizes() {
    try {
      const aspectRatio = this.video.videoWidth / this.video.videoHeight;
      const desiredWidth = window.innerWidth;

      this.aspectRatio = aspectRatio; // Store aspect ratio for later use

      // Adjust the size of the visible canvas based on the video feed dimensions
      this.displayingCanvas.width = desiredWidth;
      this.displayingCanvas.height = desiredWidth / aspectRatio;

      // Adjust processing canvas size for performance optimization
      this.adjustProcessingCanvas();
      this.frameData = new FrameData(
        this.processingCanvas.width,
        this.processingCanvas.height
      );

      // Set transformedCanvas size for the final overlay rendering
      this.transformedCanvas.width = this.displayingCanvas.width;
      this.transformedCanvas.height = this.displayingCanvas.height;
    } catch (error) {
      logError('adjustCanvasSizes', error);
    }
  }

  // Apply zoom to the video track if supported
  async applyZoomIfSupported(stream) {
    try {
      const [videoTrack] = stream.getVideoTracks();

      if (!videoTrack) {
        console.warn("No video track available");
        return;
      }

      const capabilities = videoTrack.getCapabilities();

      if ("zoom" in capabilities) {
        const minZoom = capabilities.zoom.min;
        const maxZoom = capabilities.zoom.max;
        const stepZoom = capabilities.zoom.step || 0.1;

        // Ensure desired zoom is within the allowed range
        let desiredZoom = this.desiredZoomFactor;
        if (desiredZoom < minZoom) desiredZoom = minZoom;
        if (desiredZoom > maxZoom) desiredZoom = maxZoom;

        try {
          await videoTrack.applyConstraints({
            advanced: [{ zoom: desiredZoom }],
          });
          console.log(`Applied zoom: ${desiredZoom}`);
        } catch (err) {
          console.warn("Failed to apply zoom constraints:", err);
        }
      } else {
        console.log("Zoom is not supported on this device");
      }
    } catch (error) {
      logError('applyZoomIfSupported', error);
    }
  }

  // Start video playback
  async startVideoPlayback() {
    try {
      await this.video.play();
    } catch (err) {
      console.error("Error starting video playback:", err);
      logError('startVideoPlayback', err);
      throw err;
    }
  }

  // Handle errors during camera initialization
  async handleCameraError(err) {
    try {
      // Handle specific types of errors and display appropriate messages
      if (err.name === "NotAllowedError") {
        this.displayGetCloserMessage(
          "Camera access denied. Please allow camera access in the website settings within your browser and in the system settings for your browser app."
        );
      } else if (
        err.name === "NotFoundError" ||
        err.name === "DevicesNotFoundError"
      ) {
        this.displayGetCloserMessage("No camera was found on this device.");
      } else if (
        err.name === "NotReadableError" ||
        err.name === "TrackStartError"
      ) {
        this.displayGetCloserMessage(
          "Unable to access the camera. The camera may be in use by another application."
        );
      } else if (
        err.name === "OverconstrainedError" ||
        err.name === "ConstraintNotSatisfiedError"
      ) {
        this.displayGetCloserMessage(
          "No camera matches the specified constraints. Trying the default camera."
        );

        // Fallback to default camera (front camera or system default)
        try {
          const fallbackStream = await navigator.mediaDevices.getUserMedia({
            video: true,
          });
          this.setupVideoElement(fallbackStream);
          await this.startVideoPlayback();
        } catch (fallbackErr) {
          this.displayGetCloserMessage("Fallback camera access failed.");
          console.error("Fallback camera initialization error: ", fallbackErr);
          logError('handleCameraError - fallback', fallbackErr);
        }
      } else if (err.name === "SecurityError") {
        this.displayGetCloserMessage(
          "Camera access is blocked due to security settings."
        );
      } else {
        // Generic error message for any other errors
        this.displayGetCloserMessage(
          "An unknown error occurred while initializing the camera: " + err.message
        );
      }
    } catch (error) {
      logError('handleCameraError', error);
    }
  }

  // Adjust processing canvas size based on aspect ratio
  adjustProcessingCanvas() {
    try {
      this.processingCanvas.width = this.currentProcessingCanvasWidth;
      this.processingCanvas.height =
        this.processingCanvas.width / this.aspectRatio;

      // Reassign processingContext after resizing canvas
      this.processingContext = this.processingCanvas.getContext("2d", {
        willReadFrequently: true,
      });
    } catch (error) {
      logError('adjustProcessingCanvas', error);
    }
  }

  async loadReferenceImage(url) {
    try {
      return new Promise((resolve, reject) => {
        let img = new Image();
        img.crossOrigin = "anonymous"; // Avoid cross-origin issues
        img.onload = () => {
          try {
            let tempMat = cv.imread(img);
            cv.cvtColor(tempMat, tempMat, cv.COLOR_BGRA2GRAY); // Convert to grayscale
            this.referenceMat = tempMat;
            this.referenceKeypoints = new cv.KeyPointVector();
            this.referenceDescriptors = new cv.Mat();
            // Detect features in the reference image
            this.featureDetector.detectAndCompute(
              tempMat,
              new cv.Mat(),
              this.referenceKeypoints,
              this.referenceDescriptors
            );
            resolve();
          } catch (err) {
            logError('loadReferenceImage - onload', err);
            reject(err);
          }
        };
        img.onerror = (err) => {
          logError('loadReferenceImage - img.onerror', err);
          reject(err);
        };
        img.src = url;
      });
    } catch (error) {
      logError('loadReferenceImage', error);
    }
  }

  // Load PNG frames for overlay animation
  async loadOverlayPNGs() {
    try {
      return new Promise(async (resolve, reject) => {
        try {
          let i = 1; // Start from PNG_0001.png
          this.overlayPNGs = []; // Store the loaded overlay images

          while (true) {
            const imageUrl = `${this.overlayImagePath}/PNG_${String(i).padStart(
              4,
              "0"
            )}.png`;

            // Check if the image exists by attempting to load it
            let img = new Image();
            img.crossOrigin = "anonymous";
            const imageLoaded = await new Promise((resolve) => {
              img.onload = () => resolve(true);
              img.onerror = () => resolve(false);
              img.src = imageUrl;
            });

            if (!imageLoaded) {
              break; // Stop the loop if the image doesn't exist
            }

            try {
              // Process the loaded image
              let canvas = document.createElement("canvas");
              canvas.width = img.width;
              canvas.height = img.height;
              let context = canvas.getContext("2d", {
                willReadFrequently: true,
              });

              context.drawImage(img, 0, 0);
              let mat = cv.imread(canvas); // Assuming cv.imread is available
              this.overlayPNGs.push(mat); // Push the image mat to the array
            } catch (err) {
              logError('loadOverlayPNGs - processing image', err);
            }

            i++; // Increment to load the next image
          }

          this.currentFrameIndex = 0; // Set the current frame index after loading images
          resolve();
        } catch (err) {
          logError('loadOverlayPNGs', err);
          reject(err);
        }
      });
    } catch (error) {
      logError('loadOverlayPNGs', error);
    }
  }

  // Start the frame-by-frame processing loop
  startProcessing() {
    try {
      const processFrame = async () => {
        await this.captureAndProcessFrame();
        requestAnimationFrame(processFrame); // Continuously process frames
      };
      requestAnimationFrame(processFrame);
    } catch (error) {
      logError('startProcessing', error);
    }
  }

  // Capture and process each frame from the video feed
  captureAndProcessFrame() {
    try {
      if (this.processing) return;
      this.processing = true;

      // Calculate current frame rate
      let now = Date.now();
      let elapsedTime = now - this.lastFrameTime; // in ms
      let currentFrameRate = 1000 / elapsedTime; // frames per second
      this.lastFrameTime = now; // Update last frame time

      // Adjust processing canvas width based on frame rate
      let canvasSizeChanged = false;
      let newWidth = this.currentProcessingCanvasWidth;

      if (currentFrameRate < this.targetFrameRate) {
        newWidth = Math.max(
          this.currentProcessingCanvasWidth - this.processingCanvasWidthStep,
          this.minProcessingCanvasWidth
        );
      } else if (currentFrameRate > this.targetFrameRate) {
        newWidth = this.currentProcessingCanvasWidth + this.processingCanvasWidthStep;
      }

      if (newWidth !== this.currentProcessingCanvasWidth) {
        this.currentProcessingCanvasWidth = newWidth;
        this.adjustProcessingCanvas();
        canvasSizeChanged = true;

        // Delete old frameData Mats
        if (this.frameData) {
          this.frameData.delete();
        }
        // Create new frameData with new sizes
        this.frameData = new FrameData(
          this.processingCanvas.width,
          this.processingCanvas.height
        );
      }

      // Draw the current video frame on the processing canvas
      this.processingContext.drawImage(
        this.video,
        0,
        0,
        this.processingCanvas.width,
        this.processingCanvas.height
      );

      // Extract image data from the processing canvas
      let imageData = this.processingContext.getImageData(
        0,
        0,
        this.processingCanvas.width,
        this.processingCanvas.height
      );

      // Update frame data with the new image
      this.frameData.update(imageData);

      // Detect and match features
      this.detectFeaturesAndMatch(this.frameData);

      // Now, draw the video frame on the displaying canvas
      this.displayingContext.drawImage(
        this.video,
        0,
        0,
        this.displayingCanvas.width,
        this.displayingCanvas.height
      );

      // Apply the overlay (if any)
      this.calculateTransformationAndOverlay(this.frameData);

      this.processing = false;
    } catch (error) {
      logError('captureAndProcessFrame', error);
      this.processing = false;
    }
  }

  // Detect features in the frame and match them with the reference image
  async detectFeaturesAndMatch(frameData) {
    try {
      if (!this.isOpenCVInitialized) return;

      try {
        // Detect and compute keypoints and descriptors for the current frame
        this.featureDetector.detectAndCompute(
          frameData.grayMat,
          new cv.Mat(),
          frameData.keypoints,
          frameData.descriptors
        );

        if (
          !this.referenceDescriptors.empty() &&
          !frameData.descriptors.empty()
        ) {
          // Match descriptors from current frame with the reference descriptors
          let { goodMatches, qualityIndicator } =
            this.filterMatchesWithCrossCheck(
              frameData.descriptors,
              this.referenceDescriptors
            );

          frameData.setQualityIndicator(qualityIndicator);
          frameData.goodMatches = goodMatches; // Store good matches

          // Update quality history for averaging
          this.qualityHistory.push(qualityIndicator);

          // Keep only the last nFramesForAveraging quality indicators
          if (this.qualityHistory.length > this.nFramesForAveraging) {
            this.qualityHistory.shift(); // Remove the oldest quality indicator
          }

          // Calculate the rolling average quality indicator
          const sumQuality = this.qualityHistory.reduce((sum, q) => sum + q, 0);
          this.averageQualityIndicator =
            sumQuality / this.qualityHistory.length;
          // log the average quality indicator in the div
          document.getElementById(
            "log"
          ).innerText = `Quality: ${this.averageQualityIndicator.toFixed(3)}`;
        }
      } catch (err) {
        console.error("Feature matching error: ", err);
        logError('detectFeaturesAndMatch - feature detection', err);
      }
    } catch (error) {
      logError('detectFeaturesAndMatch', error);
    }
  }

  // Filter matches using cross-checking with a distance threshold
  filterMatchesWithCrossCheck(
    srcDescriptors,
    targetDescriptors,
    distanceThreshold = this.matchDistanceThreshold // Lowe's ratio test threshold
  ) {
    try {
      let goodMatches = new cv.DMatchVector();
      let perfectMatchCount = 0;

      // Step 1: Source to Target matching
      let sourceToTargetMatches = new cv.DMatchVectorVector();
      this.matcher.knnMatch(
        srcDescriptors,
        targetDescriptors,
        sourceToTargetMatches,
        2
      );

      // Step 2: Target to Source matching
      let targetToSourceMatches = new cv.DMatchVectorVector();
      this.matcher.knnMatch(
        targetDescriptors,
        srcDescriptors,
        targetToSourceMatches,
        2
      );

      // Step 3: Cross-check matches
      for (let i = 0; i < sourceToTargetMatches.size(); ++i) {
        let match = sourceToTargetMatches.get(i);
        let dMatch1 = match.get(0); // Best match
        let dMatch2 = match.get(1); // Second-best match

        // Check for perfect matches to have a good quality indicator
        // Is set to 0.6 because the threshold can can be adapted with the variable "displayingThresholdQuality"
        if (dMatch1.distance <= dMatch2.distance * 0.75) {
          perfectMatchCount++;
        }

        // Apply Lowe's ratio test for source-to-target matches
        if (dMatch1.distance <= dMatch2.distance * distanceThreshold) {
          // Now check if the reverse match is consistent (target-to-source)
          let reverseMatch = targetToSourceMatches.get(dMatch1.trainIdx);

          // Ensure the reverse match points back to the original keypoint and apply static distance check
          if (
            reverseMatch.size() > 0 &&
            reverseMatch.get(0).trainIdx === dMatch1.queryIdx
          ) {
            goodMatches.push_back(dMatch1); // Keep the match if both directions are consistent
          }
        }
      }

      let qualityIndicator = perfectMatchCount / sourceToTargetMatches.size();
      return {
        goodMatches: goodMatches,
        qualityIndicator: qualityIndicator,
      };
    } catch (error) {
      logError('filterMatchesWithCrossCheck', error);
    }
  }

  // Calculate the transformation matrix and apply the overlay if valid
  calculateTransformationAndOverlay(frameData) {
    try {
      // Calculate opacity based on the quality indicator
      const opacity = this.calculateErrorMessageOpacity();

      // Always display the error message with the calculated opacity
      this.displayGetCloserMessage(opacity);

      // Proceed with overlay calculation if quality is good enough
      if (
        frameData.goodMatches.size() < 5 ||
        this.averageQualityIndicator < this.displayingThresholdQuality
      ) {
        // Do not apply the overlay if not enough matches or quality is below threshold
        return;
      }

      // Existing code to calculate homography and apply overlay
      let points1 = [];
      let points2 = [];

      for (let i = 0; i < frameData.goodMatches.size(); i++) {
        let match = frameData.goodMatches.get(i);
        points2.push(frameData.keypoints.get(match.queryIdx).pt.x);
        points2.push(frameData.keypoints.get(match.queryIdx).pt.y);
        points1.push(this.referenceKeypoints.get(match.trainIdx).pt.x);
        points1.push(this.referenceKeypoints.get(match.trainIdx).pt.y);
      }

      let mat1 = cv.matFromArray(points1.length / 2, 1, cv.CV_32FC2, points1);
      let mat2 = cv.matFromArray(points2.length / 2, 1, cv.CV_32FC2, points2);

      let h = cv.findHomography(mat1, mat2, cv.RANSAC);
      if (!h.empty()) {
        this.applyOverlay(h, frameData);
      }

      mat1.delete();
      mat2.delete();
      h.delete();
    } catch (error) {
      logError('calculateTransformationAndOverlay', error);
    }
  }

  calculateErrorMessageOpacity() {
    try {
      if (this.averageQualityIndicator <= this.displayingThresholdQuality) {
        return 1; // Fully opaque
      } else if (this.averageQualityIndicator >= this.qualityIndicatorMax) {
        return 0; // Fully transparent
      } else {
        // Calculate opacity using linear interpolation
        let opacity =
          (this.qualityIndicatorMax - this.averageQualityIndicator) /
          this.qualityIndicatorFadeRange;
        // Ensure opacity is between 0 and 1
        return Math.min(Math.max(opacity, 0), 1);
      }
    } catch (error) {
      logError('calculateErrorMessageOpacity', error);
    }
  }

  // Apply the overlay image to the main canvas
  applyOverlay(h, frameData) {
    try {
      if (h.empty()) return;

      // Calculate the mirrored opacity for the overlay
      const errorMessageOpacity = this.calculateErrorMessageOpacity();
      const overlayOpacity = 1 - errorMessageOpacity;

      let currentOverlayMat = this.overlayPNGs[this.currentFrameIndex];
      let transformedOverlay = new cv.Mat();

      // Warp the overlay according to the transformation matrix
      cv.warpPerspective(
        currentOverlayMat,
        transformedOverlay,
        h,
        new cv.Size(this.processingCanvas.width, this.processingCanvas.height)
      );

      // Draw the transformed overlay on the transformedCanvas
      cv.imshow(this.transformedCanvas, transformedOverlay);

      // Apply the overlay with the adjusted opacity
      this.displayingContext.globalAlpha = overlayOpacity; // Set overlay opacity
      this.displayingContext.drawImage(
        this.transformedCanvas,
        0,
        0,
        this.displayingCanvas.width,
        this.displayingCanvas.height
      );
      this.displayingContext.globalAlpha = 1; // Reset opacity to default

      transformedOverlay.delete();
      this.currentFrameIndex =
        (this.currentFrameIndex + 1) % this.overlayPNGs.length; // Cycle through overlay frames
    } catch (error) {
      logError('applyOverlay', error);
    }
  }

  // Display an error message on the canvas by darkening it and showing the message in white
  displayGetCloserMessage(opacity = 1) {
    try {
      // Darken the entire canvas with the calculated opacity
      this.displayingContext.fillStyle = `rgba(0, 0, 0, ${0.7 * opacity})`;
      this.displayingContext.fillRect(
        0,
        0,
        this.displayingCanvas.width,
        this.displayingCanvas.height
      );

      // Set message styling and opacity
      this.displayingContext.font = "15px Arial";
      this.displayingContext.fillStyle = `rgba(255, 255, 255, ${opacity})`;
      this.displayingContext.textAlign = "center";

      // Display the error message at the center of the canvas with adjusted opacity
      this.displayingContext.fillText(
        "GET CLOSER TO SEE THE MAGIC",
        this.displayingCanvas.width / 2,
        this.displayingCanvas.height / 2
      );
    } catch (error) {
      logError('displayGetCloserMessage', error);
    }
  }
}

// Window onload function to initialize AR processing
window.onload = () => {
  try {
    const referenceSrc = "HolyKingdom/MarkerSmall.jpg";
    const overlayPath = "HolyKingdom/PNG_animation_small";
    const arFeatureMatcher = new ARFeatureMatcher(
      document.getElementById("outputCanvas"),
      referenceSrc,
      overlayPath
    );

    arFeatureMatcher.initialize();
  } catch (err) {
    console.error("window.onload error: ", err);
    logError('window.onload', err);
  }
};

// Cleanup OpenCV on window unload
window.addEventListener("beforeunload", () => {
  try {
    cv.destroyAllWindows();
  } catch (error) {
    logError('beforeunload', error);
  }
});