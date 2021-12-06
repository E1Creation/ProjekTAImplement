from flask import request, jsonify
from datetime import datetime
import pyrebase
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

cred = credentials.Certificate("ta-face-recognition-firebase-adminsdk-te0ah-f0457a842d.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

todo_ref = db.collection('presensi')

firebaseConfig = {
  "apiKey": "AIzaSyD_k3FWZZ8pw4L0t8b3tNjfiqOPYg2UEJE",
  "authDomain": "ta-face-recognition.firebaseapp.com",
  "projectId": "ta-face-recognition",
  "databaseURL": "https://ta-face-recognition.firebaseapp.com",
  "storageBucket": "ta-face-recognition.appspot.com",
  "messagingSenderId": "376337421220",
  "appId": "1:376337421220:web:7ceddf910c6a80ab832731",
  "measurementId": "G-6W35NXG35T"
}

firebase = pyrebase.initialize_app(firebaseConfig)

storage = firebase.storage()

# Inserting data to firestore
def markAttendanceIntoCloud(nama):
    doc_ref = db.collection(u'presensi').document()
    dateNow = datetime.now() 
    doc_ref.set({u'nama': nama, u'waktu': dateNow })

#insertData("")
# Show firestore data to web
def ShowCloudDB():
    try:
        # Check if ID was passed to URL query
        todo_id = request.args.get('id')    
        if todo_id:
            todo = todo_ref.document(todo_id).get()
            return jsonify(todo.to_dict()), 200
        else:
            all_todos = [doc.to_dict() for doc in todo_ref.stream()]
            return jsonify(all_todos), 200
    except Exception as e:
        return f"An Error Occured: {e}"
    

