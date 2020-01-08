import smtplib
import os
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

# Email you want to send the update from (only works with gmail)
fromEmail = 'raspberrypiproject24camera@gmail.com'
# You can generate an app password here to avoid storing your password in plain text
# https://support.google.com/accounts/answer/185833?hl=en
fromEmailPassword = 'raspistill123'

# Email you want to send the update to
toEmail = 'elizabethb10124@gmail.com'

def sendEmail(nameID):
    ImgFileName = '/home/pi/opencv-face-recognition/frame.jpg'
    image = open('/home/pi/opencv-face-recognition/frame.jpg', 'rb').read()
    msgRoot = MIMEMultipart('related')
    msgRoot['Subject'] = 'Science fair'
    msgRoot['From'] = fromEmail
    msgRoot['To'] = toEmail
    msgRoot.preamble = 'Security cam found someone'
#   
#    msgText = MIMEText('<img src="cid:image1">', 'html')
#    
#    msg = 'The security camera detected: ' + nameID
#    msgText = MIMEText('The security camera detected: ' + nameID)
#    
#
#    msgRoot.attach(msgText)
 
    msgImage = MIMEImage(image, name =os.path.basename(ImgFileName))
    msgImage.add_header('Content-ID', '<image1>')
    msgRoot.attach(msgImage)
 
    smtp = smtplib.SMTP('smtp.gmail.com', 587)
    smtp.starttls()
    smtp.login(fromEmail, fromEmailPassword)
    smtp.sendmail(fromEmail, toEmail, msgRoot.as_string())
    print("Email Sent")
    smtp.quit()
