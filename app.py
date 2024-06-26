from flask import Flask, render_template, url_for, redirect, request, flash, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, LoginManager, login_user, current_user, login_required, logout_user
from flask_bcrypt import Bcrypt
import re
import os

import json
import open3d as o3d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import matplotlib
import tempfile
matplotlib.use('WebAgg')
from pcdUtils1 import read_pcd_file, main


app = Flask(__name__,template_folder='templates')
login_manager = LoginManager(app)
bcrypt = Bcrypt(app)
UPLOAD_FOLDER = './files'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SECRET_KEY'] = 'dfgtesa345Oljyfu0O'
# app.config['SESSION_COOKIE_SECURE'] = True  # Ensure cookies are only sent over HTTPS
# app.config['SESSION_COOKIE_HTTPONLY'] = True  # Mitigate XSS attacks by preventing access to cookies via JavaScript
# app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # Protect against CSRF attacks by setting SameSite attribute

db = SQLAlchemy(app)



class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(80), nullable=False)
    # is_active = db.Column(db.Boolean(), default=True)

    def __repr__(self):
        return f'<User {self.username}>'


@login_manager.user_loader
def load_user(user_id):
    user = db.session.get(User, int(user_id))
    if user:
        return user
    else:
        return None  # Or handle the missing user case differently


@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
       
        login_id = request.form['login_id']  # This can be either username or email
        password = request.form['password']
        user = User.query.filter((User.username == login_id) | (User.email == login_id)).first()

        if user:
            if bcrypt.check_password_hash(user.password, password):
                login_user(user)
                return redirect(url_for('pointCloud'))
            else:
                flash('Invalid password', 'error')
        else:
            flash('Username not found. Please register.', 'error')
            # return redirect(url_for('register'))

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
    
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        # Validate email address both
        if not re.match(r'^[a-zA-Z0-9._%+-]+@(nohara-inc\.co\.jp|gmail\.com)$', email):
            flash('Please use a valid email address ending with @nohara-inc.co.jp or @gmail.com', 'error')
            return redirect(url_for('register'))

    
        # Check if the username or email already exists in the database
        existing_user = User.query.filter_by(username=username).first()
        existing_email = User.query.filter_by(email=email).first()

        if existing_user:
            flash('Username already exists. Please choose a different one.', 'error')
        elif existing_email:
            flash('Email already exists. Please use a different email.', 'error')
        else:
            # Hash the password
            hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

            # Create a new user
            new_user = User(username=username, email=email, password=hashed_password)
            db.session.add(new_user)
            db.session.commit()

            flash('Registration successful! You can now log in.', 'success')
            return redirect(url_for('login'))


    return render_template('registeration.html')



@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        
        # Check if the email exists in the database
        user = User.query.filter_by(email=email).first()

        if user:
            # Delete the user account from the database
            db.session.delete(user)
            db.session.commit()

            flash('Your account has been deleted. Please register again.', 'success')
            return redirect(url_for('register'))
        else:
            flash('Email not found. Please enter a valid email.', 'error')

    return render_template('forgot_password.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


## deploy AI app logic from here

# key management app


@app.route('/keyManage')
@login_required
def keyManage():

    # print("User ID:", current_user.id)
    # print("Username:", current_user.username)
    # print("Email:", current_user.email)
    return render_template('keyManage.html')


# point cloud app

class DatastorePCD():
   color_list = None
   segment_plane = None
   colors = None
   org_seg = None
#    bound_box = None
#    save_pcd_path = None
   
datastorePCD = DatastorePCD() 

@app.route('/pointCloud')
@login_required
def pointCloud():
    return render_template('pointCloud.html')

@app.route('/pointCloud', methods = ['POST']) 
def uploadPCD():
   
   if 'file' not in request.files:
      return "No File Part"
   
   f = request.files['file']
   if f.filename == '':
      return 'No File Selected'
   if f:
      # read the PCD
      file_str = f.filename.split(".")[-1]
      f.filename = f"PCDmodel.{file_str}"
      f.save(os.path.join(app.config['UPLOAD_FOLDER'],f.filename))
      point_cloud_file_path = os.path.join(app.config['UPLOAD_FOLDER'],f.filename)

      # downsample
      if 'downsample' in request.form:
        pcd_raw, xyz, threshold_org = read_pcd_file(point_cloud_file_path, downsample=True)
      else : 
        pcd_raw, xyz, threshold_org = read_pcd_file(point_cloud_file_path, downsample=False)
      
      # select method
      selected_option = request.form['option']
      no_of_iters = 70000
      segment_planes, seg_color_name, seg_color_code = main(selected_option, pcd_raw, xyz, threshold_org, iterations=no_of_iters)

      datastorePCD.segment_plane = segment_planes
      datastorePCD.color_list = seg_color_name
      datastorePCD.colors = seg_color_code

      return render_template("pointCloud.html", options=datastorePCD.color_list)

@app.route('/visualizePCD', methods=['POST'])
def visualizePCD():
    # Get the list of selected options from the form
    selected_colors = request.form.getlist('selected_option')
    
    idx = [datastorePCD.color_list.index(i) for i in selected_colors] 
    selected_color_code = [datastorePCD.colors[i] for i in idx] 
    data_points = [datastorePCD.segment_plane[i] for i in idx] 

    pcd_model = o3d.geometry.PointCloud()
    for k in range(len(data_points)):
        pcd_model1 = o3d.geometry.PointCloud()
        pcd_model1.points = o3d.utility.Vector3dVector(data_points[k])
        pcd_model1.paint_uniform_color(selected_color_code[k])
        pcd_model += pcd_model1
   
    datastorePCD.org_seg = pcd_model
    num_point_visualize = 4000000
    
    xyz = np.asarray(pcd_model.points)
    rgb = np.asarray(pcd_model.colors)
    if len(xyz) > num_point_visualize:
        # pcd_model = pcd_model.voxel_down_sample(0.05)
        k_points = int(np.round(len(xyz) / num_point_visualize))
        pcd_model = pcd_model.uniform_down_sample(every_k_points = k_points)
        xyz = np.asarray(pcd_model.points)
        rgb = np.asarray(pcd_model.colors)
    
    points_with_colors = np.hstack((xyz, rgb))
    point_cloud_data = points_with_colors.tolist()

    return render_template("pointCloud.html", options=datastorePCD.color_list, point_cloud_data=json.dumps(point_cloud_data))


@app.route('/download_point_cloud', methods=['GET'])
def download_point_cloud():
   file_format = request.args.get('file_format')
   if file_format == 'txt':
      # Save the point cloud to a temporary XYZ file
      points = np.asarray(datastorePCD.org_seg.points)
      colors = np.asarray(datastorePCD.org_seg.colors)
      points_with_colors = np.hstack((points, colors))

      temp_xyz_file = tempfile.NamedTemporaryFile(suffix='.txt', delete=False)
      np.savetxt(temp_xyz_file.name, points_with_colors, fmt='%.3f', delimiter=' ')
      temp_xyz_file.close()


      # Serve the temporary XYZ file for download
      return send_file(temp_xyz_file.name, as_attachment=True, download_name='point_cloud.txt')

   elif file_format == 'ply':
      # Save the point cloud to a temporary PLY file
      temp_ply_file = tempfile.NamedTemporaryFile(suffix='.ply', delete=False)
      o3d.io.write_point_cloud(temp_ply_file.name, datastorePCD.org_seg)
      temp_ply_file.close()

      # Serve the temporary PLY file for download
      return send_file(temp_ply_file.name, as_attachment=True, download_name='point_cloud.ply')

    
@app.route('/export_excel', methods=['GET'])
def export_excel():

   # Create an ExcelWriter object with xlsxwriter engine
   excel_file_path = 'files/Area_color_segments.xlsx'
   writer = pd.ExcelWriter(excel_file_path, engine='xlsxwriter')

   with writer:
      # Create a workbook and worksheet objects
      workbook = writer.book
      worksheet = workbook.add_worksheet()

      # Write column headers
      worksheet.write(0, 0, 'Color_name')
      # worksheet.write(0, 1, 'Color_code')
      worksheet.write(0, 1, 'R')
      worksheet.write(0, 2, 'G')
      worksheet.write(0, 3, 'B')
      worksheet.write(0, 4, 'Area')
      worksheet.write(0, 5, 'x_min')
      worksheet.write(0, 6, 'x_max')
      worksheet.write(0, 7, 'y_min')
      worksheet.write(0, 8, 'y_max')
      worksheet.write(0, 9, 'z_min')
      worksheet.write(0, 10, 'z_max')
      worksheet.write(0, 11, 'Image')

      for i in range(len(datastorePCD.segment_plane)-1):
         data_points = datastorePCD.segment_plane[i]

         ranges = np.ptp(data_points, axis=0)
         min_index = np.argmin(ranges)  # get minimum range value index
         # delete the minimum range coordinate value
         plot_data = np.delete(data_points, min_index, 1)
         x_range = [plot_data[:, 0].min(), plot_data[:, 0].max()]
         y_range = [plot_data[:, 1].min(), plot_data[:, 1].max()]

         plt.figure(figsize=(5,5))
         plt.scatter(plot_data[:, 0], plot_data[:, 1], s=1, c=[datastorePCD.colors[i]], marker='o')
         plt.xlabel('X')
         plt.ylabel('Y')
         plt.title(f"{datastorePCD.color_list[i]}, {datastorePCD.colors[i]}")
         plt.xlim(x_range)
         plt.ylim(y_range)

         savefig_path = os.path.join(app.config['UPLOAD_FOLDER'],f'contour_image{i}.png') # '.\static\images\contour_image.png'
         plt.savefig(savefig_path)
         

         image = cv2.imread(savefig_path) 

         # Convert to grayscale 
         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
         # Blur the image 
         blurred = cv2.GaussianBlur(gray, (5, 5), 0) 
         thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
         # Detect edges 
         edges = cv2.Canny(blurred, 50, 150) 
         # Find contours 
         contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
         # Filter contours 
         rects = [] 
         for contour in contours: 
            # Approximate the contour to a polygon 
            polygon = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True) 
            
            # Check if the polygon has 4 sides and the aspect ratio is close to 1 
            if len(polygon) == 4 and abs(1 - cv2.contourArea(polygon) / (cv2.boundingRect(polygon)[2] * cv2.boundingRect(polygon)[3])) < 0.1: 
               rects.append(polygon) 

         # Find the largest rectangle
         largest_rect = max(rects, key=cv2.contourArea)
         # Get the bounding box of the largest rectangle
         x, y, w, h = cv2.boundingRect(largest_rect)

         cropped_image = thresh[y:y+h-2, x:x+w-2]
         invert = cv2.bitwise_not(cropped_image)
         num_0 = np.sum(invert == 0)
         num_255 = np.sum(invert == 255)

         ratio = num_0 / (num_255 + num_0)
         area = ratio * (x_range[1] - x_range[0]) * (y_range[1] - y_range[0])

         min_values = np.round(data_points.min(axis=0), 3)
         max_values = np.round(data_points.max(axis=0), 3)


         # Write name to first column
         worksheet.write(i + 1, 0, datastorePCD.color_list[i])
         # Write surname to second column
         # worksheet.write(i + 1, 1, data.colors[i])
         worksheet.write(i + 1, 1, datastorePCD.colors[i][0])
         worksheet.write(i + 1, 2, datastorePCD.colors[i][1])
         worksheet.write(i + 1, 3, datastorePCD.colors[i][2])
         worksheet.write(i + 1, 4, np.round(area,5))
         worksheet.write(i + 1, 5, min_values[0])
         worksheet.write(i + 1, 6, max_values[0])
         worksheet.write(i + 1, 7, min_values[1])
         worksheet.write(i + 1, 8, max_values[1])
         worksheet.write(i + 1, 9, min_values[2])
         worksheet.write(i + 1, 10, max_values[2])

        #  # Calculate image row based on current data row
        #  image_row = i + 1  # + 1 for header row

         # Insert image in third column (adjust column index as needed)
         worksheet.insert_image(i + 1, 11, savefig_path)

         plt.cla()
         plt.close()

   return send_file(excel_file_path, as_attachment=True, download_name='Area_color_segments.xlsx')
  








@app.route('/floorGAN')
@login_required
def floorGAN():

    return render_template('floorGAN.html')

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    # app.run(host='127.0.0.1', port=8080, debug=True)
    app.run(host='0.0.0.0', port=8080, debug=True)
    # app.run(host='0.0.0.0', port=8080, threaded=True)
