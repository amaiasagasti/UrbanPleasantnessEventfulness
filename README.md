


---
<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/github_username/repo_name">
    <img src="data/images/logo.png" alt="Logo">
  </a>

<h3 align="center">SOUNDLIGHTS</h3>

  <p align="center">
    Research and development of AI models to predict the perceptual sound qualities of "pleasantness" and "eventfulness" in urban soundscapes with focus on prediction accuracy and suitability to real-time implementations. Additionally, an example code of a simulation is included: a recorded audio file simulates an audio being captured in real time, pleasantness and eventfulness values are calculated every few seconds.
  </p>
  
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
<p align="left">
    This project is a colaboration between the following institutions:
    <p>
    <img src="data/images/MTG_logo.jpeg" alt="Logo" height=60>    
    </p>
    <p>
    <img src="data/images/Bitlab_logo.png" alt="Logo" height=60>
    </p>
    Funded by:
    <p>
    <img src="data/images/Logo-Bithabitat.png" alt="Logo" height=60>
    </p>
  </p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>




<!-- GETTING STARTED -->
## Getting Started
This section provides all the necessary information to set up the working environment The followiing bullet-list details the set up process:
- It is recommended to use ```Python==3.10.14``` to guarantee compatibility.
- Install all the depencencies detailed in the ```requirements.txt``` file of this project. 
- <a href="https://github.com/LAION-AI/CLAP">LAION-AI's CLAP model</a> is used, make sure you install its dependencies too.


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

To test the developed models for predicting the pleasantness (P) and eventfulness (E) values of an urban soundscape, two scripts have been implemented and are ready to run. These scripts simulate the functionality of an acoustic environment sensor by processing a WAV file containing an urban soundscape recording. The simulation operates in "real time," where the audio file is read continuously, and every X seconds, the scripts output the predicted values of P and E. This setup allows for the real-time analysis and evaluation of the acoustic environment based on the provided recordings.

The first script is ```src/scripts/sensor_simulate.py```. To run this script, the function ```play_audio()``` is called:
```
play_audio(
    path,
    frequency,
    duration,
    maintain_time,
)
```
The first input is the path to the audio file that will simulate a real-time recording. The second input specifies the frequency, in seconds, at which new predictions for pleasantness (P) and eventfulness (E) will be made. This also determines the duration of the audio slices that will be saved locally. The third parameter defines the duration, in seconds, of audio that will be considered for prediction calculations. This duration must be an integer multiple of the frequency, allowing an exact number of audio slices to be concatenated. Finally, the fourth input sets the duration, in seconds, for which the audio slices will be stored locally. This value should be as small as possible but must be greater than or equal to the "duration" parameter and also an integer multiple of the "frequency. 

This function saves the predicted values of P and E in a text file, with each new pair of values written on a new line.

The second script, src/scripts/sensor_read_and_plot.py, operates by reading the most recent line from the aforementioned text file in real-time and plotting the values on a simple interface.

<img src="data/images/simulation_capture.png" alt="Logo">

<p></p>
A demo of this simulation can be found <a href="link!!!">here </a>.
<p></p>
If you wish to integrate the predictions into your own code, you can find the models for predicting pleasantness and eventfulness in the /models/trained directory. These models require LAION-AI's CLAP embeddings of the input audio.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



## Reproducibility

### Generation of the dataset ARAUS-extended

<img src="data/images/dataset-schema.png" alt="Logo">

<img src="data/images/model-schema.png" alt="Logo">

<img src="data/images/sensor-schema.png" alt="Logo">

<img src="data/images/legend.png" alt="Logo">


1) Preliminary adaptation
   ---

    Data needed: responses.csv (from ARAUS original dataset)
   
    Script to run: `dataset_Adequate_ARAUS_for_extension.ipynb`

    Result: responses_Soundlights.csv
    
    ---

2) Generate dataset ARAUS-extended
3) Generate fold0 variations
4) Generate fold-Fs - listening test analysis

### Overview
Our project aims to [brief description of the project's long-term goals and vision].


### Contributors' Guide

We welcome contributions to help achieve these milestones. Please check out our [contributing guide](CONTRIBUTING.md) for more details on how to get involved.


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Your Name - [@twitter_handle](https://twitter.com/twitter_handle) - email@email_client.com

Project Link: [https://github.com/github_username/repo_name](https://github.com/github_username/repo_name)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* []()
* []()
* []()

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo_name/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo_name/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo_name/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo_name/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
