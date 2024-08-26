from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser 
from urllib.error import HTTPError

"""
check if allowed, check if not, check if exists
"""


def can_scrape(url):

    rp = RobotFileParser()
    parsed = urlparse(url)
    robot_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp.set_url(robot_url)

    try:
        rp.read()
        return rp.can_fetch('*', url), rp.crawl_delay('*')
    
    except HTTPError as e:
        #no robots.txt
        return e.code == 404, rp.crawl_delay('*')
    
    except: 
        #invalid url or error due to network, 403, etc
        return False, None

print(can_scrape(""))