# OpenAI with Assistant API and Knowledge Retreival

First run `python app.py` and then yun the `streamlit_app.py` using command `streamlit run streamlit_app.py`. Once the interface is running on the `8501` port upload the research paper you want the AI to help you with. Make sure it is less than 20 pages so that the OpenAI generates the response to the query quickly. Once uploaded chat functionality will be unlocked. You can now ask any questions against that research paper.

![alt text](image.png)

# Running Locally

To simply run the chatbot application locally on our PC:

```bash
pip install -r requirements
streamlit run streamlit_app.py
```

# Running on Docker

Below are the instructions to build and run the container. Once we build and run the container we could visit `http://0.0.0.0:8501` or `http://localhost:8501` to access the chatbot application. 

## Build a Docker image

The docker build command builds an image from a Dockerfile. Run the following command from the main application directory on your server to build the image:

```bash
docker build -t streamlit .
```

The -t flag is used to tag the image. Here, we have tagged the image streamlit. If you run:

```bash
docker images
```
You should see a streamlit image under the REPOSITORY column. For example:

```bash
REPOSITORY   TAG       IMAGE ID       CREATED              SIZE
streamlit    latest    70b0759a094d   About a minute ago   1.02GB
```

## Run the Docker container

Now that you have built the image, you can run the container by executing:

```bash
docker run -p 8501:8501 streamlit
```

The -p flag publishes the container’s port 8501 to your server’s 8501 port.

If all went well, you should see an output similar to the following:

```bash
docker run -p 8501:8501 streamlit

  You can now view your Streamlit app in your browser.

  URL: http://0.0.0.0:8501
```

To view your app, users can browse to http://0.0.0.0:8501 or http://localhost:8501