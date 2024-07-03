from django.test import TestCase

from userLogin.models import User


# Create your tests here.
class TestUserRegistration(TestCase):
    def setUp(self):
        self.user = User.objects.create(
            username='cxh',
            password='123456',
            email = '1111',
            phone = '1111',
            type = 'cccc'
        )
    def test_user_creation(self):
        self.assertEqual(self.user.username, 'cxh')