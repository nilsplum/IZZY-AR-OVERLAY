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


    //The higher, the more features are detected which leads to more matches but also more false positives
    this.featureDetectionSensitivity = 0.0005;
    // The larger, the more accurate the matches but also the more time needs the method "this.featureDetector.detectAndCompute()" which is the most time consuming method
    this.processingCanvasFixedWidth = 500;

    //PERFORMANCE AND QUALITY SETTINGS
    // Configurable canvas size for processing
    // Indirectly how many features are detected
    // Features are possible matches
    this.matchDistanceThreshold = 0.75; // Lowe's ratio test threshold for cross-checking
    // Initialize essential properties

    // For blocking the overlay when quality is poor
    this.matchQualityIndicatorValue = 0.6; // Threshold for quality indicator
    this.displayingThresholdQuality = 0.002; // Minimum quality indicator for displaying overlay

    //Best for far away: MarkerSmall.jpg

    //JPG Markers work better than PNG -> Because of the blur?



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

    //PNG Set animation images and current frame index
    this.overlayPNGs = [];
    this.currentFrameIndex = 0;

    // Initialisation
    this.isOpenCVInitialized = false;
    this.processing = false;

    //Current frame data
    this.frameData = null;

    // Off-screen canvas for lower resolution processing
    this.processingCanvas = document.createElement("canvas");
    this.processingContext = this.processingCanvas.getContext("2d", {
      willReadFrequently: true,
    });
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
  async initializeCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" },
      });

      this.video = document.createElement("video");
      this.video.setAttribute("playsinline", "true");
      this.video.muted = true;
      this.video.style.display = "none"; // Hide video element
      document.body.appendChild(this.video);

      this.video.srcObject = stream;

      this.video.onloadedmetadata = () => {
        const aspectRatio =
          this.video.videoWidth / this.video.videoHeight;
        const desiredWidth = window.innerWidth;

        // Adjust main visible canvas size
        this.displayingCanvas.width = desiredWidth;
        this.displayingCanvas.height = desiredWidth / aspectRatio;

        // Adjust processing canvas size for performance
        this.adjustProcessingCanvas(aspectRatio);
        this.frameData = new FrameData(
          this.processingCanvas.width,
          this.processingCanvas.height
        );

        // Set transformedCanvas size for final overlay rendering
        this.transformedCanvas.width = this.displayingCanvas.width;
        this.transformedCanvas.height = this.displayingCanvas.height;
      };

      await this.video.play();
    } catch (err) {
      console.error("Camera initialization error: ", err);
    }
  }

  // Adjust processing canvas size based on aspect ratio
  adjustProcessingCanvas(aspectRatio) {
    this.processingCanvas.width = this.processingCanvasFixedWidth;
    this.processingCanvas.height =
      this.processingCanvas.width / aspectRatio;
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
      const cacheBuster = `?cb=${new Date().getTime()}`;
      const imageUrl = `${this.overlayImagePath}/PNG_${String(i).padStart(
        4,
        "0"
      )}.png${cacheBuster}`;

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

    // Always display the video feed on the main canvas
    this.displayingContext.drawImage(
      this.video,
      0,
      0,
      this.displayingCanvas.width,
      this.displayingCanvas.height
    );

    this.calculateTransformationAndOverlay(this.frameData);

    this.processing = false;
  }

  // Detect features in the frame and match them with the reference image
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

      // Perform feature matching if both reference and frame descriptors are available
      if (
        !this.referenceDescriptors.empty() &&
        !frameData.descriptors.empty()
      ) {
        // Match descriptors from current frame (source) with the reference descriptors (target)
        let { goodMatches, qualityIndicator } =
          this.filterMatchesWithCrossCheck(
            frameData.descriptors,
            this.referenceDescriptors
          );

        frameData.setQualityIndicator(qualityIndicator);
        // Store good matches in the frameData
        frameData.goodMatches = goodMatches;

        // Log the number of good matches
        //console.log(`Number of good matches: ${goodMatches.size()}`);
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
      if (
        dMatch1.distance <=
        dMatch2.distance * this.matchQualityIndicatorValue
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

    // Log details about the filtering process
    /*
    console.log(
      `Total source-to-target matches: ${sourceToTargetMatches.size()}`
    );
    console.log(
      `Good matches after cross-checking: ${goodMatches.size()}`
    );
    */

    let qualityIndicator =
      perfectMatchCount / sourceToTargetMatches.size();
    return {
      goodMatches: goodMatches,
      qualityIndicator: qualityIndicator,
    };
  }

  // Calculate the transformation matrix and apply the overlay if valid
  calculateTransformationAndOverlay(frameData) {
    if (
      frameData.goodMatches.size() < 5 ||
      frameData.qualityIndicator < this.displayingThresholdQuality
    ) {
      this.displayErrorMessage("GET CLOSER TO SEE THE MAGIC");
      return;
    }
    //alert(frameData.qualityIndicator);

    let points1 = [];
    let points2 = [];

    // Extract matching keypoints from both reference and current frame
    for (let i = 0; i < frameData.goodMatches.size(); i++) {
      let match = frameData.goodMatches.get(i);
      points2.push(frameData.keypoints.get(match.queryIdx).pt.x);
      points2.push(frameData.keypoints.get(match.queryIdx).pt.y);
      points1.push(this.referenceKeypoints.get(match.trainIdx).pt.x);
      points1.push(this.referenceKeypoints.get(match.trainIdx).pt.y);
    }

    // Convert keypoint arrays to Mats
    let mat1 = cv.matFromArray(
      points1.length / 2,
      1,
      cv.CV_32FC2,
      points1
    );
    let mat2 = cv.matFromArray(
      points2.length / 2,
      1,
      cv.CV_32FC2,
      points2
    );

    // Find the homography matrix (transformation matrix)
    let h = cv.findHomography(mat1, mat2, cv.RANSAC);
    if (!h.empty()) {
      // Apply overlay
      this.applyOverlay(h, frameData);
    }
    mat1.delete();
    mat2.delete();
    h.delete();
  }

  // Compute the aspect ratio of a transformed quadrilateral
  computeAspectRatio(corners) {
    let width = Math.sqrt(
      Math.pow(corners.data32F[2] - corners.data32F[0], 2) +
        Math.pow(corners.data32F[3] - corners.data32F[1], 2)
    );
    let height = Math.sqrt(
      Math.pow(corners.data32F[6] - corners.data32F[0], 2) +
        Math.pow(corners.data32F[7] - corners.data32F[1], 2)
    );
    return width / height;
  }

  // Compute the area of a transformed quadrilateral
  computeQuadrilateralArea(corners) {
    let pts = [];
    for (let i = 0; i < 4; i++) {
      pts.push({
        x: corners.data32F[i * 2],
        y: corners.data32F[i * 2 + 1],
      });
    }
    // Shoelace formula for computing polygon area
    let area =
      0.5 *
      Math.abs(
        pts[0].x * pts[1].y +
          pts[1].x * pts[2].y +
          pts[2].x * pts[3].y +
          pts[3].x * pts[0].y -
          pts[1].x * pts[0].y -
          pts[2].x * pts[1].y -
          pts[3].x * pts[2].y -
          pts[0].x * pts[3].y
      );
    return area;
  }

  // Apply the overlay image to the main canvas
  applyOverlay(h, frameData) {
    if (h.empty()) return;

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

    // Blend the transformed overlay with the main canvas
    this.displayingContext.drawImage(
      this.transformedCanvas,
      0,
      0,
      this.displayingCanvas.width,
      this.displayingCanvas.height
    );

    transformedOverlay.delete();
    this.currentFrameIndex =
      (this.currentFrameIndex + 1) % this.overlayPNGs.length; // Cycle through overlay frames
  }

  // Display an error message on the canvas by darkening it and showing the message in white
  displayErrorMessage(message) {
    // Darken the entire canvas
    this.displayingContext.fillStyle = "rgba(0, 0, 0, 0.7)";
    this.displayingContext.fillRect(
      0,
      0,
      this.displayingCanvas.width,
      this.displayingCanvas.height
    );

    // Set message styling
    this.displayingContext.font = "15px Arial";
    this.displayingContext.fillStyle = "white";
    this.displayingContext.textAlign = "center";

    // Display the error message at the center of the canvas
    this.displayingContext.fillText(
      message,
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