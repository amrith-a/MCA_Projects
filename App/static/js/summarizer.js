
window.onload = function() {
    localStorage.setItem('fileType','none')
    localStorage.setItem('step',0)
  };



function changeFile(input) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();

        reader.onload = function (e) {
            document.querySelector('.filename').textContent = input.value.replace(/^.*\\/, "")
            // $('.chosen-img')
            //     .attr('src', e.target.result);
            document.querySelector('.next').style.userSelect = 'auto'
            document.querySelector('.next').style.opacity = '1'
            document.querySelector('.next').style.pointerEvents = 'auto'
        };

        reader.readAsDataURL(input.files[0]);
    }
}


function enableNext(){
    
        data = document.querySelector('.textData').value
        if (data.length < 0 ){
            alert("Please enter the values")
        }
        else{
            document.querySelector('.next').style.userSelect = 'auto'
            document.querySelector('.next').style.opacity = '1'
            document.querySelector('.next').style.pointerEvents = 'auto'
        }
}

function selectFile(fileType){

    if (fileType == 'pdf'){

        document.querySelector('.pdf').classList.add('active-file');
        document.querySelector('.text').classList.remove('active-file');

        localStorage.setItem('fileType','pdf');
        document.querySelector('.next').style.userSelect = 'auto'
        document.querySelector('.next').style.opacity = '1'
        document.querySelector('.next').style.pointerEvents = 'auto'
        
    }

    else if (fileType == 'text'){
        document.querySelector('.pdf').classList.remove('active-file');
        document.querySelector('.text').classList.add('active-file');

        localStorage.setItem('fileType','text');
        document.querySelector('.next').style.userSelect = 'auto'
        document.querySelector('.next').style.opacity = '1'
        document.querySelector('.next').style.pointerEvents = 'auto'

    }
}

function nextStep(){

    var step = localStorage.getItem('step')
    if (step == 0){

        localStorage.setItem('step',1);

        document.querySelector('.select-file').style.display = 'none';
        
        if(localStorage.getItem('fileType') == 'pdf'){
            document.querySelector('.upload-file').style.display = 'flex';
        }
        else{
            document.querySelector('.text-file').style.display = 'flex';
        }
        
        document.querySelector('.stepone').classList.remove('active');
        document.querySelector('.steptwo').classList.add('active');
        document.querySelector('.prev').style.opacity = '0.5'
        document.querySelector('.prev').style.userSelect = 'auto'
        document.querySelector('.prev').style.pointerEvents = 'auto'

        document.querySelector('.next').style.userSelect = 'none'
        document.querySelector('.next').style.opacity = '0.5'
        document.querySelector('.next').style.pointerEvents = 'none'

        

    }
    else if(step == 1){
        localStorage.setItem('step',2);
        document.querySelector('.upload-file').style.display = 'none';
        document.querySelector('.text-file').style.display = 'none';

        document.querySelector('.summary').style.display = 'flex';

        document.querySelector('.stepthree').classList.add('active');
        document.querySelector('.steptwo').classList.remove('active');

        document.querySelector('.next').style.userSelect = 'none'
        document.querySelector('.next').style.opacity = '0.5'
        document.querySelector('.next').style.pointerEvents = 'none'

        document.querySelector('.prev').style.opacity = '1'
        if(localStorage.getItem('fileType') == 'pdf'){
            sendData();
        }
        else{
            sendTextData();
            
        }
        
        
    }


}


function prevStep(){

    var step = localStorage.getItem('step');
    if (step == 1){
        localStorage.setItem('step',0);

        document.querySelector('.select-file').style.display = 'flex';
        
        if(localStorage.getItem('fileType') == 'pdf'){
            document.querySelector('.upload-file').style.display = 'none';
        }
        else{
            document.querySelector('.text-file').style.display = 'none';
        }

        document.querySelector('.text-file').style.display = 'none';
        document.querySelector('.stepone').classList.add('active');
        
        document.querySelector('.steptwo').classList.remove('active');

        document.querySelector('.prev').style.opacity = '0.3'
        document.querySelector('.prev').style.userSelect = 'none'
        document.querySelector('.prev').style.pointerEvents = 'none'
        
    }

    else if(step == 2){
        localStorage.setItem('step',1);

        
        if(localStorage.getItem('fileType') == 'pdf'){
            document.querySelector('.upload-file').style.display = 'flex';
        }
        else{
            document.querySelector('.text-file').style.display = 'flex';
        }
        document.querySelector('.stepthree').classList.remove('active');
        document.querySelector('.summary').style.display = 'none';
        document.querySelector('.steptwo').classList.add('active');
        
        document.querySelector('.next').style.userSelect = 'auto'
        document.querySelector('.next').style.opacity = '1'
        document.querySelector('.next').style.pointerEvents = 'auto'

        document.querySelector('.prev').style.opacity = '0.5'
        document.querySelector('.prev').style.userSelect = 'auto'
        document.querySelector('.prev').style.pointerEvents = 'auto'

    }

    

}

$(function() {

    var mainbottom = $('#main').offset().top + $('#main').height()/25;
   
    $(window).on('scroll',function(){

        stop = Math.round($(window).scrollTop());
     
        if (stop > mainbottom) {
            $(".navbar").css('background-color', '#141414');
        } else {
            $('.navbar').css('background-color', 'transparent');
        }
    });
});