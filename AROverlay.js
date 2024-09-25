class FrameData {
  constructor(width, height) {
    // Initialize reusable Mats for downscaled processing resolution
    this.srcMat = new cv.Mat(height, width, cv.CV_8UC4);
    this.grayMat = new cv.Mat(height, width, cv.CV_8UC1);
    this.keypoints = new cv.KeyPointVector();
    this.descriptors = new cv.Mat();
    this.goodMatches = new cv.DMatchVector();
    this.transformationMatrix = null;
    this.qualityIndicator = null;
  }

  update(imageData) {
    // Update srcMat with new image data
    this.srcMat.data.set(imageData.data);

    // Convert to grayscale for feature detection
    cv.cvtColor(this.srcMat, this.grayMat, cv.COLOR_BGRA2GRAY);

    // Clear and reinitialize keypoints and descriptors
    this.keypoints = new cv.KeyPointVector();
    this.descriptors.delete();
    this.descriptors = new cv.Mat();
    this.goodMatches = new cv.DMatchVector(); // Clear matches
  }

  setQualityIndicator(qualityValue) {
    this.qualityIndicator = qualityValue;
  }
  // Clean up Mats when no longer needed
  delete() {
    this.srcMat.delete();
    this.grayMat.delete();
    this.keypoints.delete();
    this.descriptors.delete();
    this.goodMatches.delete();
  }
}

class ARFeatureMatcher {
  constructor(canvas, referenceImageUrl, overlayImagePath) {
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
  }

  async initialize() {
    try {
      // Initialize OpenCV and camera, load reference images
      await this.initializeOpenCV();
      await this.setupCVDependentProperties();
      await this.initializeCamera();
      await this.loadReferenceImage(this.referenceImageUrl);
      await this.loadOverlayPNGs();
      this.startProcessing();
    } catch (err) {
      console.error("Initialization error: ", err);
    }
  }

  // Asynchronously initialize OpenCV.js
  initializeOpenCV() {
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
  }

  // Set up OpenCV-dependent properties like feature detector and matcher
  async setupCVDependentProperties() {
    if (this.isOpenCVInitialized) {
      this.featureDetector = new cv.AKAZE();
      this.featureDetector.setThreshold(this.featureDetectionSensitivity); // Adjust detection sensitivity
      this.matcher = new cv.BFMatcher(); // Matcher for feature matching
    } else {
      console.error("OpenCV not initialized.");
    }
  }

  // Initialize the camera stream and adjust canvas dimensions based on the video feed
  // Initialize the camera stream and adjust canvas dimensions based on the video feed
  async initializeCamera() {
    try {
      // Check if camera APIs are supported
      await this.checkCameraSupport();

      // Get the camera stream
      const stream = await this.getCameraStream();

      // Set up the video element with the obtained stream
      this.setupVideoElement(stream);

      // Apply zoom if supported
      await this.applyZoomIfSupported(stream);

      // Start video playback
      await this.startVideoPlayback();
    } catch (err) {
      console.error("Camera initialization error: ", err);

      // Handle errors and execute default behavior
      await this.handleCameraError(err);
    }
  }

  // Check if navigator.mediaDevices and getUserMedia are supported
  async checkCameraSupport() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      this.displayErrorMessage(
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
        this.displayErrorMessage(
          "Camera access has been denied. Please enable it in your browser settings."
        );
        throw new Error("Camera access denied");
      }
    } catch (err) {
      // Some browsers may not support navigator.permissions
      console.warn("Permissions API not supported, proceeding without checking permissions.");
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
      throw err;
    }
  }

  // Set up the video element with the obtained stream
  setupVideoElement(stream) {
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
    };
  }

  // Adjust canvas sizes based on video dimensions
  adjustCanvasSizes() {
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
  }

  // Apply zoom to the video track if supported
  async applyZoomIfSupported(stream) {
    const [videoTrack] = stream.getVideoTracks();

    if (!videoTrack) {
      console.warn("No video track available");
      return;
    }

    const capabilities = videoTrack.getCapabilities();

    if ('zoom' in capabilities) {
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
        // Proceed with default behavior without zoom
      }
    } else {
      console.log("Zoom is not supported on this device");
      // Proceed with default behavior without zoom
    }
  }

  // Start video playback
  async startVideoPlayback() {
    try {
      await this.video.play();
    } catch (err) {
      console.error("Error starting video playback:", err);
      throw err;
    }
  }

  // Handle errors during camera initialization
  async handleCameraError(err) {
    // Handle specific types of errors and display appropriate messages
    if (err.name === "NotAllowedError") {
      this.displayErrorMessage(
        "Camera access denied. Please allow camera access in the website settings within your browser and in the system settings for your browser app."
      );
    } else if (
      err.name === "NotFoundError" ||
      err.name === "DevicesNotFoundError"
    ) {
      this.displayErrorMessage("No camera was found on this device.");
    } else if (
      err.name === "NotReadableError" ||
      err.name === "TrackStartError"
    ) {
      this.displayErrorMessage(
        "Unable to access the camera. The camera may be in use by another application."
      );
    } else if (
      err.name === "OverconstrainedError" ||
      err.name === "ConstraintNotSatisfiedError"
    ) {
      this.displayErrorMessage(
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
        this.displayErrorMessage("Fallback camera access failed.");
        console.error("Fallback camera initialization error: ", fallbackErr);
      }
    } else if (err.name === "SecurityError") {
      this.displayErrorMessage(
        "Camera access is blocked due to security settings."
      );
    } else {
      // Generic error message for any other errors
      this.displayErrorMessage(
        "An unknown error occurred while initializing the camera: " + err.message
      );
    }
  }

  // Adjust processing canvas size based on aspect ratio
  adjustProcessingCanvas() {
    this.processingCanvas.width = this.currentProcessingCanvasWidth;
    this.processingCanvas.height =
      this.processingCanvas.width / this.aspectRatio;

    // Reassign processingContext after resizing canvas
    this.processingContext = this.processingCanvas.getContext("2d", {
      willReadFrequently: true,
    });
  }

  // Load the reference image for feature matching
  async loadReferenceImage(url) {
    let img = new Image();
    img.crossOrigin = "anonymous"; // Avoid cross-origin issues
    img.src = url;

    await new Promise((resolve, reject) => {
      img.onload = () => {
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
      };
      img.onerror = reject;
    });
  }

  // Load PNG frames for overlay animation
  async loadOverlayPNGs() {
    let i = 1; // Start from PNG_0001.png
    this.overlayPNGs = []; // Store the loaded overlay images

    const iframe = document.createElement("iframe");
    iframe.style.display = "none";
    document.body.appendChild(iframe);

    while (true) {
      const imageUrl = `${this.overlayImagePath}/PNG_${String(i).padStart(
        4,
        "0"
      )}.png`;

      // Use iframe to check if the image exists
      const imageExists = await new Promise((resolve) => {
        const iframeDoc =
          iframe.contentDocument || iframe.contentWindow.document;
        const testImg = iframeDoc.createElement("img");
        testImg.onload = () => resolve(true); // Image exists
        testImg.onerror = () => resolve(false); // Image doesn't exist
        testImg.src = imageUrl;
      });

      if (!imageExists) {
        break; // If the image doesn't exist, stop the loop
      }

      // Load the image as usual into the main document if it exists
      let img = new Image();
      img.crossOrigin = "anonymous";
      img.src = imageUrl;

      const imageLoaded = await new Promise((resolve) => {
        img.onload = () => {
          let canvas = document.createElement("canvas");
          canvas.width = img.width;
          canvas.height = img.height;
          let context = canvas.getContext("2d", {
            willReadFrequently: true,
          });

          context.drawImage(img, 0, 0);
          let mat = cv.imread(canvas); // Assuming cv.imread is available
          this.overlayPNGs.push(mat); // Push the image mat to the array
          resolve(true); // Image loaded successfully
        };

        img.onerror = () => {
          resolve(false); // If the image load fails, stop the loop silently
        };
      });

      if (!imageLoaded) {
        break; // Stop the loop if loading the image fails
      }

      i++; // Increment to load the next image
    }

    iframe.remove(); // Clean up the iframe after the loop is done
    this.currentFrameIndex = 0; // Set the current frame index after loading images
  }

  // Start the frame-by-frame processing loop
  startProcessing() {
    const processFrame = async () => {
      await this.captureAndProcessFrame();
      requestAnimationFrame(processFrame); // Continuously process frames
    };
    requestAnimationFrame(processFrame);
  }

  // Capture and process each frame from the video feed
  captureAndProcessFrame() {
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
  }

  // Detect features in the frame and match them with the reference image
  async detectFeaturesAndMatch(frameData) {
    if (!this.isOpenCVInitialized) return;
  
    try {
      // Detect and compute keypoints and descriptors for the current frame
      this.featureDetector.detectAndCompute(
        frameData.grayMat,
        new cv.Mat(),
        frameData.keypoints,
        frameData.descriptors
      );
  
      if (!this.referenceDescriptors.empty() && !frameData.descriptors.empty()) {
        // Match descriptors from current frame with the reference descriptors
        let { goodMatches, qualityIndicator } = this.filterMatchesWithCrossCheck(
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
        this.averageQualityIndicator = sumQuality / this.qualityHistory.length;
        // log the average quality indicator in the div
        document.getElementById("log").innerText = `Quality: ${this.averageQualityIndicator.toFixed(3)}`;
      }
    } catch (err) {
      console.error("Feature matching error: ", err);
    }
  }
  

  // Filter matches using cross-checking with a distance threshold
  filterMatchesWithCrossCheck(
    srcDescriptors,
    targetDescriptors,
    distanceThreshold = this.matchDistanceThreshold // Lowe's ratio test threshold
  ) {
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
      if (
        dMatch1.distance <=
        dMatch2.distance * 0.75
      ) {
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
  }

  // Calculate the transformation matrix and apply the overlay if valid
  calculateTransformationAndOverlay(frameData) {
    // Calculate opacity based on the quality indicator
    const opacity = this.calculateErrorMessageOpacity();

    // Always display the error message with the calculated opacity
    this.displayErrorMessage(opacity);

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
  }

  calculateErrorMessageOpacity() {
    if (this.averageQualityIndicator <= this.displayingThresholdQuality) {
      return 1; // Fully opaque
    } else if (this.averageQualityIndicator >= this.qualityIndicatorMax) {
      return 0; // Fully transparent
    } else {
      // Calculate opacity using linear interpolation
      let opacity = (this.qualityIndicatorMax - this.averageQualityIndicator) / this.qualityIndicatorFadeRange;
      // Ensure opacity is between 0 and 1
      return Math.min(Math.max(opacity, 0), 1);
    }
  }

  // Apply the overlay image to the main canvas
  applyOverlay(h, frameData) {
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
      new cv.Size(
        this.processingCanvas.width,
        this.processingCanvas.height
      )
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
  }
  

  // Display an error message on the canvas by darkening it and showing the message in white
  displayErrorMessage(opacity=1) {
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
  }
};

// Cleanup OpenCV on window unload
window.addEventListener("beforeunload", () => {
  cv.destroyAllWindows();
});
