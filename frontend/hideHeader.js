export function hideHeader() {
    let lastScrollPosition = 0;
    const header = document.querySelector('header');
    const headerTab = document.getElementById('headerTab');
    const navItem = document.querySelectorAll('.navbar a')

    window.addEventListener('scroll', () => {
        const currentScrollPosition = window.scrollY;
        if (currentScrollPosition != lastScrollPosition && currentScrollPosition != header.offsetHeight) {
            header.style.transform = 'translateY(-100%)';
        }
        lastScrollPosition = currentScrollPosition;
    });

    headerTab.addEventListener('click', () => {
        header.style.transform = 'translateY(0%)';
    });

    navItem.addEventListener('click', () => {
        header.style.transform = 'translateY(-100%)';
    });

  }


  
  