let currentImageIndex = 0;
const images = document.getElementsByClassName("image");

function showImage(index) {
  if (index < 0) {
    index = images.length - 1;
  } else if (index >= images.length) {
    index = 0;
  }

  for (let i = 0; i < images.length; i++) {
    images[i].style.display = "none";
  }

  images[index].style.display = "block";
  currentImageIndex = index;
}

function changeImage(direction) {
  currentImageIndex += direction;
  showImage(currentImageIndex);
}

showImage(currentImageIndex);