from threading import Thread
from flask_mail import Message
from .. import app, mail


def send_async_email(msg):
    with app.app_context():
        mail.send(msg)

def send_email(to, subject, html):
    msg = Message(app.config['MAIL_PREFIX'] + subject,
                  sender=app.config['MAIL_USERNAME'],
                  recipients=[to])
    msg.html = html
    thr = Thread(target=send_async_email, args=[msg])
    thr.start()

    return thr
