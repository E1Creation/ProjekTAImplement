from typing import List
from flask import render_template
import os
import numpy as np
import face_recognition as fr
import csv
import cv2
from datetime import datetime
from statistics import mode
from firestore import markAttendanceIntoCloud
import mysql.connector

# Database
# 1 Koneksi DB
db = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="face_recognition"
)

if db.is_connected():
  print("Berhasil terhubung ke database")
mycursor=db.cursor()

# Import Gambar
path = 'static/assets/uploads'
csvPath = 'static/assets/csv/daftarhadir.csv'
gambar = []
listNama = []
listname = []
myList = os.listdir(path) 
for cls in myList:
    curImg = cv2.imread(f'{path}/{cls}') # membaca semua gambar di direktori
    gambar.append(curImg) # menambahkan semua gambar ke list gambar
    listNama.append(os.path.splitext(cls)[0]) # menambahkan nama file ke dalam list listNama

# baca file csv
with open('static/assets/csv/encodeFace.csv', 'r') as f:
  file = csv.reader(f)
  encodeKnownFace = list(file)

def insertRow(id, nama, tanggal, jam, status):
    mycursor=db.cursor()
    sqlInput="REPLACE into rlabkom2(id, nama, tanggal, jam, status) values (%s,%s,%s,%s,%s)"
    data=(id, nama, tanggal, jam, status)
    mycursor.execute(sqlInput, data)
    db.commit()
    print("Data User yang telah presensi sudah ditambahkan pada tabel rlabkom2 database face_recognition")
  
def markAttendanceIntoDB(id, nama):
    now = datetime.now()
    tanggal =now.strftime('%d-%m-%y')
    jam = now.strftime('%H:%M:%S')
    status ="masuk"
    insertRow(id, nama, tanggal, jam, status)
    print("User yang melakukan presensi telah terdata")

def ShowRiwayat():
    mycursor = db.cursor()
    sql = "SELECT * FROM rlabkom2"
    mycursor.execute(sql)
    result = mycursor.fetchall()
    mycursor.close()
    return render_template('riwayat.html', riwayat=result)

def gen_frame():
    listname=[]
    listid=[]
    camera = cv2.VideoCapture(0)
    print("Kamera telah menyala")
    while True:
        try:
                success, frame = camera.read()
                frameSmall = cv2.resize(frame, (0,0), None, 0.25, 0.25)
                frameSmall = cv2.cvtColor(frameSmall, cv2.COLOR_BGR2RGB) # Convert ke Grescale

                faceLocWeb = fr.face_locations(frameSmall) # Menemukan lokasi wajah dalam list berisi 4 koordinat
                encodeWebcam = fr.face_encodings(frameSmall, faceLocWeb) # Encode gambar yg ditangkap dr webcam
                # Membandingkan wajah 
                for encodeFace, faceLoc in zip(encodeWebcam, faceLocWeb):
                    matches = fr.compare_faces(encodeKnownFace, encodeFace) # membandingkan webcam dgn daftar wajah yg sudah dikenali
                    faceDist = fr.face_distance(encodeKnownFace, encodeFace)
                    # print(faceDist)   
                    matchesIndex = np.argmin(faceDist)

                    if matches[matchesIndex]:
                        nama = listNama[matchesIndex]
                        nama = str(nama)
                        # Membuat kotak hijau dgn nama dibawahnya
                        y1, x2, y2, x1 = faceLoc
                        y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                        # cv2.rectangle(frame, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
                        cv2.putText(frame, nama, (x1+6, y2-6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        print(nama)
                        listname.append(nama)
                        listid.append(str(matchesIndex))
                        print(len(listname))
                        print(listname)
                        if len(listname) > 5:
                            markAttendanceIntoDB(mode(listname), mode(listid))
                            markAttendanceIntoCloud(listname)
                            print("Daftar kehadiran berhasil")
                            listname.clear()
                            
        except Exception as e:
            print(str(e))
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
        if cv2.waitKey(1) & 0xFF ==ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()