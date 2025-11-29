document.addEventListener('DOMContentLoaded', () => {
    const slides = document.querySelectorAll('.slide');
    const prevBtn = document.getElementById('prev-btn');
    const nextBtn = document.getElementById('next-btn');

    let currentSlideIndex = 0;

    // Function to show a specific slide
    function showSlide(index) {
        // Stop at the boundaries (no wrapping)
        if (index >= slides.length) {
            currentSlideIndex = slides.length - 1;
        } else if (index < 0) {
            currentSlideIndex = 0;
        } else {
            currentSlideIndex = index;
        }

        // Remove active class from all slides
        slides.forEach(slide => {
            slide.classList.remove('active');
            // Also ensure inline styles for title slide are reset if needed, 
            // though class removal should handle the z-index if CSS is set up right.
            // But since we used !important inline, we might need to handle that if it persists.
            // However, the inline style is on the element itself, not added by JS.
            // The issue is likely the z-index: 100 !important on the title slide 
            // staying effective if the class isn't enough or if it overrides others.
            // Actually, the inline style is PERMANENT on that element.
            // We need to hide it when it's not active.
            if (slide.classList.contains('title-slide')) {
                slide.style.opacity = ''; 
                slide.style.zIndex = '';
            }
        });

        // Add active class to current slide
        const currentSlide = slides[currentSlideIndex];
        currentSlide.classList.add('active');
        
        // Re-apply inline styles for title slide if it's the current one
        if (currentSlide.classList.contains('title-slide')) {
            currentSlide.style.opacity = '1';
            currentSlide.style.zIndex = '100';
        }

        // Reset animation for Thank You slide
        if (currentSlide.classList.contains('thank-you-slide')) {
            const text1 = currentSlide.querySelector('.thank-you-text');
            const text2 = currentSlide.querySelector('.questions-text');
            
            // Reset first text
            if (text1) {
                text1.style.animation = 'none';
                text1.offsetHeight; /* trigger reflow */
                text1.style.animation = null; 
            }

            // Reset second text
            if (text2) {
                text2.style.animation = 'none';
                text2.offsetHeight; /* trigger reflow */
                text2.style.animation = null; 
            }
        }
    }

    // Next Slide
    function nextSlide() {
        showSlide(currentSlideIndex + 1);
    }

    // Previous Slide
    function prevSlide() {
        showSlide(currentSlideIndex - 1);
    }

    // Start Presentation (Enter Fullscreen on first click)
    function enterFullScreen() {
        const elem = document.documentElement;
        if (elem.requestFullscreen) {
            elem.requestFullscreen().catch(err => {
                console.log(`Error attempting to enable full-screen mode: ${err.message} (${err.name})`);
            });
        } else if (elem.webkitRequestFullscreen) { /* Safari */
            elem.webkitRequestFullscreen();
        } else if (elem.msRequestFullscreen) { /* IE11 */
            elem.msRequestFullscreen();
        }
        
        // Remove listener after first successful click
        document.removeEventListener('click', enterFullScreen);
    }

    document.addEventListener('click', enterFullScreen);

    // Keyboard Navigation
    document.addEventListener('keydown', (e) => {
        // Don't navigate if user is typing in an input field
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') {
            return;
        }

        if (e.key === 'ArrowRight' || e.key === ' ') {
            nextSlide();
        } else if (e.key === 'ArrowLeft') {
            prevSlide();
        }
    });

    // Button Navigation
    nextBtn.addEventListener('click', nextSlide);
    prevBtn.addEventListener('click', prevSlide);

    // Prediction Form Logic
    const predictionForm = document.getElementById('predictionForm');
    if (predictionForm) {
        predictionForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Get form values
            const store = parseFloat(document.getElementById('store').value);
            const dept = parseFloat(document.getElementById('dept').value);
            const temp = parseFloat(document.getElementById('temperature').value);
            const fuel = parseFloat(document.getElementById('fuelPrice').value);
            const cpi = parseFloat(document.getElementById('cpi').value);
            const unemployment = parseFloat(document.getElementById('unemployment').value);
            const storeSize = parseFloat(document.getElementById('storeSize').value);
            const isHoliday = parseInt(document.getElementById('isHoliday').value);
            const month = parseInt(document.getElementById('month').value);
            
            // Simple prediction formula (demo only)
            let baseSales = storeSize * 0.15;
            
            // Adjust for temperature
            baseSales *= (1 + (temp - 60) * 0.002);
            
            // Adjust for fuel price (inverse relationship)
            baseSales *= (1 - (fuel - 3) * 0.05);
            
            // Adjust for CPI
            baseSales *= (cpi / 200);
            
            // Adjust for unemployment (inverse)
            baseSales *= (1 - unemployment * 0.01);
            
            // Holiday boost
            if (isHoliday === 1) {
                baseSales *= 1.5;
            }
            
            // Seasonal adjustment
            const seasonalFactors = [0.9, 0.85, 0.95, 1.0, 1.05, 1.1, 1.15, 1.1, 1.0, 1.05, 1.2, 1.4];
            baseSales *= seasonalFactors[month - 1];
            
            // Department factor
            baseSales *= (1 + dept * 0.01);
            
            // Display result
            const resultBox = document.getElementById('resultBox');
            const resultValue = document.getElementById('resultValue');
            
            resultValue.textContent = '$' + baseSales.toFixed(2).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
            resultBox.style.display = 'block';
            
            // Scroll to result within the slide
            resultBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        });
    }
    // Mouse Cursor Hiding Logic for Fullscreen
    let mouseTimer = null;
    const cursorHideDelay = 2000; // 2 seconds

    function hideCursor() {
        document.body.classList.add('hide-cursor');
    }

    function showCursor() {
        document.body.classList.remove('hide-cursor');
    }

    function onMouseMove() {
        showCursor();
        if (mouseTimer) {
            clearTimeout(mouseTimer);
        }
        // Only set timer if we are still in fullscreen
        if (document.fullscreenElement || document.webkitFullscreenElement || document.msFullscreenElement) {
            mouseTimer = setTimeout(hideCursor, cursorHideDelay);
        }
    }

    function handleFullscreenChange() {
        const isFullscreen = document.fullscreenElement || document.webkitFullscreenElement || document.msFullscreenElement;

        if (isFullscreen) {
            // "Hidden suddenly" on enter
            hideCursor();
            document.addEventListener('mousemove', onMouseMove);
        } else {
            // Cleanup on exit
            showCursor();
            document.removeEventListener('mousemove', onMouseMove);
            if (mouseTimer) clearTimeout(mouseTimer);
        }
    }

    document.addEventListener('fullscreenchange', handleFullscreenChange);
    document.addEventListener('webkitfullscreenchange', handleFullscreenChange);
    document.addEventListener('msfullscreenchange', handleFullscreenChange);

    // Code Viewer Logic
    const viewCodeBtn = document.getElementById('viewCodeBtn');
    const codeModal = document.getElementById('codeModal');
    const closeCodeBtn = document.getElementById('closeCodeBtn');
    const codeContent = document.getElementById('codeContent');

    if (viewCodeBtn && codeModal && closeCodeBtn && codeContent) {
        viewCodeBtn.addEventListener('click', (e) => {
            e.stopPropagation(); // Prevent slide navigation
            codeContent.textContent = typeof PROJECT_CODE !== 'undefined' ? PROJECT_CODE : 'Error: Code content not found.';
            codeModal.classList.add('active');
        });

        closeCodeBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            codeModal.classList.remove('active');
        });

        // Close on click outside
        codeModal.addEventListener('click', (e) => {
            if (e.target === codeModal) {
                codeModal.classList.remove('active');
            }
        });
        
        // Prevent clicks inside modal from closing it or navigating slides
        codeModal.querySelector('.code-container-wrapper').addEventListener('click', (e) => {
            e.stopPropagation();
        });
    }
});
