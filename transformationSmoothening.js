class ComprehensiveSmoothnessFilter {
    constructor(windowSize = 5, alpha = 0.1, outlierThreshold = 0.5) {
        this.slidingWindowAveragingFilter = new SlidingWindowAveragingFilter(windowSize);
        this.emaFilter = new ExponentialMovingAverageFilter(alpha);
        this.outlierDetection = new OutlierDetection(outlierThreshold);
    }

    addHomography(h) {
        if (!this.outlierDetection.isOutlier(h)) {
            this.slidingWindowAveragingFilter.addHomography(h);
        }
    }

    getFilteredHomography() {
        try {
            let avgH = this.slidingWindowAveragingFilter.getAverageHomography();
            if (avgH) {
                return this.emaFilter.filter(avgH);
            }
            return null;
        } catch (e) {
            console.error("ComprehensiveSmoothnessFilter.getFilteredHomography: ", e);
            return null;
        }
    }
}
class SlidingWindowAveragingFilter {
    constructor(windowSize = 5) {
        this.windowSize = windowSize;
        this.homographyBuffer = [];
    }

    addHomography(h) {
        if (this.homographyBuffer.length >= this.windowSize) {
            this.homographyBuffer.shift();
        }
        this.homographyBuffer.push(h.clone());
    }

    getAverageHomography() {
        try {
            if (this.homographyBuffer.length === 0) {
                return null;
            }

            let sumH = new cv.Mat(this.homographyBuffer[0].rows, this.homographyBuffer[0].cols, cv.CV_64F, new cv.Scalar(0));

            for (let h of this.homographyBuffer) {
                cv.add(sumH, h, sumH);
            }

            let avgH = new cv.Mat(sumH.rows, sumH.cols, sumH.type());
            let bufferSize = this.homographyBuffer.length;

            for (let i = 0; i < sumH.rows; i++) {
                for (let j = 0; j < sumH.cols; j++) {
                    avgH.doublePtr(i, j)[0] = sumH.doublePtr(i, j)[0] / bufferSize;
                }
            }
            sumH.delete();

            return avgH;
        } catch (e) {
            console.error("SlidingWindowAveragingFilter.getAverageHomography: ", e);
            return null;
        }
    }
}
class OutlierDetection {
    constructor(threshold = 0.5) {
        this.threshold = threshold;
        this.previousHomography = null;
    }

    isOutlier(h) {
        try {
            if (!this.previousHomography) {
                this.previousHomography = h.clone();
                return false;
            }

            let diff = new cv.Mat();
            cv.absdiff(this.previousHomography, h, diff);

            let totalDiff = 0;
            for (let i = 0; i < diff.rows; i++) {
                for (let j = 0; j < diff.cols; j++) {
                    totalDiff += Math.abs(diff.doublePtr(i, j)[0]);
                }
            }

            diff.delete();

            if (totalDiff > this.threshold) {
                return true;
            } else {
                this.previousHomography.delete();
                this.previousHomography = h.clone();
                return false;
            }
        } catch (e) {
            console.error("OutlierDetection.isOutlier: ", e);
            return false;
        }
    }
}
class ExponentialMovingAverageFilter {
    constructor(alpha = 0.1) {
        this.alpha = alpha;
        this.emaHomography = null;
    }

    filter(h) {
        try {
            if (!this.emaHomography) {
                this.emaHomography = h.clone();
                return h;
            }

            let smoothedH = new cv.Mat(h.rows, h.cols, h.type());
            for (let i = 0; i < h.rows; i++) {
                for (let j = 0; j < h.cols; j++) {
                    smoothedH.doublePtr(i, j)[0] = this.alpha * h.doublePtr(i, j)[0] + (1 - this.alpha) * this.emaHomography.doublePtr(i, j)[0];
                }
            }

            this.emaHomography.delete();
            this.emaHomography = smoothedH.clone();

            return smoothedH;
        } catch (e) {
            console.error("ExponentialMovingAverageFilter.filter: ", e);
            return null;
        }
    }
}
