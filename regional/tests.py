from django.test import TestCase
import sys
# Create your tests here.
from regional import detect0406
from regional.detect0406 import Main

if __name__ == "__main__":
    app = detect0406.QApplication(sys.argv)
    win = Main()
    win.show()
    sys.exit(app.exec_())