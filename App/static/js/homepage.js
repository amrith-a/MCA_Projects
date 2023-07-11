

function scrollToAbout(){
   
    var element = document.getElementById("about");
    
    element.scrollIntoView();
}

function scrollToTeam(){
    var element = document.getElementById("team");
    
    element.scrollIntoView();
}

function showLogin(){
    document.querySelector('.login-wrapper').style.display = "flex";
    document.querySelector('.signup-wrapper').style.display = "none";
}

function closeLogin(){
    document.querySelector('.login-wrapper').style.display = "none";
}

function showSignup(){
    document.querySelector('.signup-wrapper').style.display = "flex";
    document.querySelector('.login-wrapper').style.display = "none";
}

function closeSignup(){
    document.querySelector('.signup-wrapper').style.display = "none";
}

$(function() {

    var mainbottom = $('#main').offset().top + $('#main').height()/12;
   
    $(window).on('scroll',function(){

        stop = Math.round($(window).scrollTop());
     
        if (stop > mainbottom) {
            $(".navbar").css('background-color', '#141414');
        } else {
            $('.navbar').css('background-color', 'transparent');
        }
    });
});


window.addEventListener('load',function() {

    document.querySelector('.navbar').style.top = "0";
    
    
})

