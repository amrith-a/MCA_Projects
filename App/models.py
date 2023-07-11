from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class userProfile(models.Model):

    user = models.ForeignKey(User,on_delete=models.CASCADE)
    
    image = models.ImageField(upload_to='profile_pic',default='sherlock.jpg')
    
    def __str__(self):
        return self.user.username + " profile"
    
class fileUpload(models.Model):

    # user = models.ForeignKey(User,on_delete = models.CASCADE)
    file = models.FileField(upload_to = 'media/file_uploads')

    def __str__(self):
        return self.user.username + "'s file"