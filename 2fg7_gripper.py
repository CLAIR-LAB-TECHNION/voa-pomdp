import pycurl
import xmlrpc.client
from io import BytesIO

# inspired from here: https://github.com/RyanPaulMcKenna/onRobot/blob/main/onRobot/onRobot/gripper.py

class Gripper2FG7:
    def __init__(self, robot_ip, gripper_id):
        self.gripper_id = gripper_id
        self.robot_ip = robot_ip

    def set_gripper_ext(self, width: float=0, force: float=50, speed: float=100) -> bool:
        xml_request = f"""<?xml version="1.0"?>
    <methodCall>
    <methodName>twofg_grip_ext</methodName>
        <params>
            <param>
                <value><int>{self.gripper_id}</int></value>
            </param>
            <param>
                <value><double>{width}</double></value>
            </param>
            <param>
                <value><double>{force}</double></value>
            </param>
            <param>
                <value><double>{speed}</double></value>
            </param>
        </params>
    </methodCall>"""

        headers = ["Content-Type: application/x-www-form-urlencoded"]

        # headers = ["User-Agent: Python-PycURL", "Accept: application/json"]
        data = xml_request.replace('\r\n','').encode()
        # Create a new cURL object
        curl = pycurl.Curl()

        # Set the URL to fetch
        curl.setopt(curl.URL, f'http://{self.robot_ip}:41414')
        curl.setopt(curl.HTTPHEADER, headers)
        curl.setopt(curl.POSTFIELDS, data)
        # Create a BytesIO object to store the response
        buffer = BytesIO()
        curl.setopt(curl.WRITEDATA, buffer)

        # Perform the request
        curl.perform()

        # Get the response body
        response = buffer.getvalue()

        # Print the response
        print(response.decode('utf-8'))

        # Close the cURL object
        curl.close()

if __name__ == "__main__":
    gripper = Gripper2FG7("192.168.0.10", 0)
    gripper.set_gripper_ext(0, 50, 100)

