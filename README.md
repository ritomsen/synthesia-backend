# synthesia-backend

pip install -r requirements.txt 

uvicorn backend:app --reload

curl -X POST \
     -F "image=@backend/<IMAGE_NAME>" \
     http://localhost:8000/describe-image-musically
