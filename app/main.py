import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main():
    logger = logging.getLogger(__name__)
    logger.info("Hello from python-uv-template!")


if __name__ == "__main__":
    main()
