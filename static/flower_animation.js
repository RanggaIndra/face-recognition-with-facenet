/**
 * Flower animation for Lia's face recognition app
 * This script handles the creation and animation of flower elements
 * when Lia is detected by the face recognition system.
 */

// Animation interval reference
let animationInterval;

/**
 * Creates and displays a single animated flower
 */
function showRoseAnimation() {
    const roseContainer = document.getElementById("rose-container");
    if (!roseContainer) return;

    const rose = document.createElement("div");
    rose.classList.add("rose");

    // Create flower petals
    for (let j = 0; j < 4; j++) {
        const petal = document.createElement("div");
        petal.classList.add("petal");
        rose.appendChild(petal);
    }

    // Flower center
    const center = document.createElement("div");
    center.classList.add("center");
    rose.appendChild(center);

    // Flower stem
    const stem = document.createElement("div");
    stem.classList.add("stem");
    rose.appendChild(stem);

    // Random horizontal position
    rose.style.left = Math.random() * 90 + "%";

    roseContainer.appendChild(rose);

    // Remove rose after animation completes
    setTimeout(() => {
        rose.remove();
    }, 6000);
}

/**
 * Starts the flower animation by showing initial flowers
 * and setting up an interval for continuous animation
 */
function startRoseAnimation() {
    // Clear any existing interval
    stopRoseAnimation();

    // Show initial burst of flowers
    for (let i = 0; i < 5; i++) {
        setTimeout(showRoseAnimation, i * 200);
    }

    // Continue showing flowers every second
    animationInterval = setInterval(showRoseAnimation, 1000);
}

/**
 * Stops the ongoing flower animation
 */
function stopRoseAnimation() {
    if (animationInterval) {
        clearInterval(animationInterval);
        animationInterval = null;
    }
}

/**
 * Handles the animation based on detection result
 * @param {boolean} isDetected - Whether Lia was detected
 */
function handleRoseAnimationForResult(isDetected) {
    const roseContainer = document.getElementById("rose-container");

    if (isDetected) {
        roseContainer.style.display = "block";
        startRoseAnimation();
    } else {
        roseContainer.style.display = "none";
        stopRoseAnimation();
    }
}