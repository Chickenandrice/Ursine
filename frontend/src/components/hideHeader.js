export function hideHeader() {
    let lastScrollPosition = 0;
    const header = document.querySelector('header');
    const headerTab = document.getElementById('headerTab');

    window.addEventListener('scroll', () => {
        const currentScrollPosition = window.scrollY;
        if (currentScrollPosition > header.offsetHeight){
            if (currentScrollPosition != lastScrollPosition) {
                header.style.transform = 'translateY(-100%)';
            }
        }
        lastScrollPosition = currentScrollPosition;
    });

    headerTab.addEventListener('click', () => {
        header.style.transform = 'translateY(0%)';
    });
  }


  
  