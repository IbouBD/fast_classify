# celery_config.py
from celery import Celery

def make_celery(app):
    celery = Celery(
        app.import_name,
        backend=app.config['result_backend'],
        broker=app.config['broker_url']
    )
    celery.conf.update(app.config)
    celery.conf.update(
        worker_prefetch_multiplier=1,  # Précharge un seul message à la fois
        task_acks_late=True,  # Accusé de réception tardif
    )
    return celery
