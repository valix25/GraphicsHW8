
#include <iostream>
#include <fstream>
#include <string>
#include <array>
//#include <SDL2/SDL.h>
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/freeglut.h>
#include <ply.hpp>
#include <limits.h>
#include <unistd.h>
#include <math.h>
#include "include/Polynomial.hpp"
#include "include/lodepng.h"
#include "include/plyloader.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/surface_matching.hpp"
#include <opencv2/surface_matching/ppf_helpers.hpp>
#include <opencv2/surface_matching/ppf_match_3d.hpp>
#include "opencv2/viz.hpp"

#define RAY_TRACE_DEPTH 2

typedef std::array<unsigned int, 4> uColorRGBA;
typedef std::array<float, 4> fColorRGBA;
typedef std::vector<std::vector<uColorRGBA> > ImageContainer;

struct Point3D {
	double x;
	double y;
	double z;
	unsigned long index;
	std::array<double, 3> n;
	Point3D()
	{
		n[0] = n[1] = n[2] = 0.0;
	}
};

struct Vec3D {
	double x;
	double y;
	double z;
};

struct Triangle {
	Point3D p;
	Point3D q;
	Point3D r;
	std::array<double, 3> face_n;
	Triangle()
	{
		face_n[0] = face_n[1] = face_n[2] = 0.0;
	}
};

typedef std::vector<Triangle> Object;

struct DataObject{
	Object obj;
	cv::Mat faces;
	cv::Mat points;
	bool ifAlreadyScaledTranslated;
	bool hasMirroring;
	int refractionIndex;
	DataObject()
	{
		ifAlreadyScaledTranslated = false;
		hasMirroring = false;
		refractionIndex = 1;
	}
};

DataObject bunny1;
DataObject bunny2;
DataObject buddha1;

std::string title = "GlobalGL";
const unsigned int PIXELS_HEIGHT = 600;
const unsigned int PIXELS_WIDTH = 600;

const unsigned int PIX_H = 100;
const unsigned int PIX_W = 100;

/* viewpoint data */
float x_pos = 0;
float y_pos = 1;
float z_pos = 2;
float theta = M_PI/6;
float phi = M_PI/2;
float R = 2.25;
float at_x = 0, at_y = 0, at_z = 0;
float up_x = 0, up_y = 1.0, up_z = 0;

/* perspective data */
float perspective_angle = 90.0;
float perspective_ratio = 1.0;
float perspective_near = 1.0;
float perspective_far = 1000.0;

/* background color */
fColorRGBA background = {0.0f, 0.15f, 0.5f, 1.0f};

/* blinn-phong parameters */
float light_x = 0;
float light_y = 5;
float light_z = 7;
float light_extra = 0;
fColorRGBA Ia = {1, 1, 1, 1};
fColorRGBA ka = {0.5, 0, 0, 1};
fColorRGBA Ip = {0.5, 0.5, 0.5, 1};
fColorRGBA kd = {0.8, 0.8, 0.8, 1.0};
fColorRGBA ks = {1, 1, 1, 1};
float s_exp = 50;
// parameters for the atenuation factor
// default attenuation parameters in glLightfv
float c0 = 1;
float c1 = 0;
float c2 = 0;

// the window resizing callback function
void changeSize(int w, int h);
// the keyboard callback function
void processKeys(unsigned char key, int x, int y);
// the drawing callback function
void display();

// Get current date/time, format is YYYY-MM-DD.HH:mm:ss
const std::string currentDateTime() {
	time_t     now = time(0);
	struct tm  tstruct;
	char       buf[80];
	tstruct = *localtime(&now);
	strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);
	return buf;
}

uColorRGBA transformColor(const fColorRGBA& col)
{
	uColorRGBA new_col = {0,0,0,255};
	for(unsigned int i = 0; i < 4; i++)
	{
		float f2 = std::max(0.0, std::min(1.0, (double) col[i]));
		new_col[i] = floor((f2 == 1.0) ? 255 : f2 * 256.0);
	}
	return new_col;
}

void encodePNG(ImageContainer& image, unsigned int &cols,
	unsigned int &rows, std::string filepath)
{
	std::vector<unsigned char> imageRaw;
	for(unsigned int i = 0; i < rows; i++)
		for(unsigned int j = 0; j < cols; j++)
		{
			imageRaw.push_back(image[i][j][0]);
			imageRaw.push_back(image[i][j][1]);
			imageRaw.push_back(image[i][j][2]);
			imageRaw.push_back(image[i][j][3]);
		}

	unsigned error = lodepng::encode(filepath, imageRaw, cols, rows, LCT_RGBA);
	if(error)
	{
		std::cerr << "encoder error " << error << ": ";
		std::cerr << lodepng_error_text(error) << "\n";
	}
}

bool checkIntersection(const float& x_pos,const float& y_pos, 
	const float& z_pos, const float& s_x, const float& s_y, 
	const float& s_z, const Triangle& t)
{
	float v1_x, v1_y, v1_z, v2_x, v2_y, v2_z;
	for(unsigned int i = 0; i < 3; i++)
	{
		if(i == 0)
		{
			v1_x = t.p.x - x_pos;
			v1_y = t.p.y - y_pos;
			v1_z = t.p.z - z_pos;
			v2_x = t.q.x - x_pos;
			v2_y = t.q.y - y_pos;
			v2_z = t.q.z - z_pos;
		} else if(i == 1){
			v1_x = t.q.x - x_pos;
			v1_y = t.q.y - y_pos;
			v1_z = t.q.z - z_pos;
			v2_x = t.r.x - x_pos;
			v2_y = t.r.y - y_pos;
			v2_z = t.r.z - z_pos;			
		} else {
			v1_x = t.r.x - x_pos;
			v1_y = t.r.y - y_pos;
			v1_z = t.r.z - z_pos;
			v2_x = t.p.x - x_pos;
			v2_y = t.p.y - y_pos;
			v2_z = t.p.z - z_pos;		
		}
		float n_x, n_y, n_z;
		n_x = v2_y * v1_z - v2_z * v1_y;
		n_y = v2_z * v1_x - v2_x * v1_z;
		n_z = v2_x * v1_y - v2_y * v1_x;
		float n_length = sqrt(n_x * n_x + n_y * n_y + n_z * n_z);
		n_x = n_x / n_length;
		n_y = n_y / n_length;
		n_z = n_z / n_length;
		float d = -(x_pos * n_x + y_pos * n_y + z_pos * n_z);
		if ((s_x * n_x + s_y * n_y + s_z * n_z + d) < 0) return false;
	}
	return true;
}

float getTofIntersection(const float& x_pos,const float& y_pos, 
	const float& z_pos, const float& s_x, const float& s_y, 
	const float& s_z, const Triangle& tri)
{
	// t = -(P0 * N + d) / (V * N) -- lecture notes
	float vn = s_x * tri.face_n[0] + s_y * tri.face_n[1] + s_z * tri.face_n[2];
	float p0n = x_pos * tri.face_n[0] + y_pos * tri.face_n[1] + z_pos * tri.face_n[2];
	// P * N + d = 0
	float d = -(tri.p.x * tri.face_n[0] + tri.p.y * tri.face_n[1] + tri.p.z * tri.face_n[2]);
	return -(p0n + d) / vn; 
}

fColorRGBA computeRayTrace(float x_pos, float y_pos, float z_pos,
	float s_x, float s_y, float s_z, unsigned int depth)
{
	fColorRGBA col = {0,0,0,1};
	std::vector<DataObject > d_objects;
	d_objects.push_back(bunny1);
	d_objects.push_back(bunny2);
	d_objects.push_back(buddha1);
	float min_t; bool first_t = true;
	Triangle nearest_tri; 
	unsigned int nearest_obj;
	for(unsigned int i = 0; i < d_objects.size(); i++)
	{
		for(unsigned int j = 0; j < d_objects[i].obj.size(); j++)
		{
			Triangle tri = d_objects[i].obj[j];
			bool check = checkIntersection(x_pos, y_pos, z_pos, s_x, s_y, s_z, tri);
			if(check) 
			{	
				float t = getTofIntersection(x_pos, y_pos, z_pos, s_x, s_y, s_z, tri);
				if(first_t)
				{
					min_t = t;
					nearest_obj = i;
					nearest_tri = tri;
					first_t = false;
				} else {
					if(t < min_t)
					{
						min_t = t;
						nearest_obj = i;
						nearest_tri = tri;
					}
				}
			}
		}
	}
	// first_t remains true only if there is no intersection
	if(first_t) return background;
	else 
	{
		float i_x = x_pos + min_t * s_x;
		float i_y = y_pos + min_t * s_y;
		float i_z = z_pos + min_t * s_z;

		return col;
	}
}

void computeRayTraceWrapper()
{
	ImageContainer image;

	image.resize(PIX_H+1);
	for(unsigned int i = 0; i < PIX_H+1; i++)
	{
		image[i].resize(PIX_W+1);
		for(unsigned int j = 0; j < PIX_W+1; j++)
		{
			image[i][j][0] = 0;
			image[i][j][1] = 0;
			image[i][j][2] = 0;
			image[i][j][3] = 255;
		}
	}

	float view_vec_x = at_x - x_pos;
	float view_vec_y = at_y - y_pos;
	float view_vec_z = at_z - z_pos;
	float view_vec_len = sqrt(view_vec_x*view_vec_x + view_vec_y* view_vec_y 
					+ view_vec_z * view_vec_z);
	float n_v_x = view_vec_x / view_vec_len;
	float n_v_y = view_vec_y / view_vec_len;
	float n_v_z = view_vec_z / view_vec_len;

	/* get frustum's near plane corner coordinates */
	float Hnear = 2 * tan(perspective_angle / 2) * perspective_near;
	float Wnear = Hnear * perspective_ratio;

	float up_length = sqrt(up_x*up_x + up_y*up_y + up_z*up_z);
	float n_up_x = up_x / up_length;
	float n_up_y = up_y / up_length;
	float n_up_z = up_z / up_length;

	// cross product: n_v x n_up
	float right_x = n_v_y * n_up_z - n_v_z * n_up_y;
	float right_y = n_v_z * n_up_x - n_v_x * n_up_z;
	float right_z = n_v_x * n_up_y - n_v_y * n_up_x;

	// near plane center point from viewpoint
	float nc_x = x_pos + n_v_x * perspective_near;
	float nc_y = y_pos + n_v_y * perspective_near;
	float nc_z = z_pos + n_v_z * perspective_near;

	unsigned int image_i = 0, image_j = 0;
	for(int i = (int) PIX_H / 2; i >= -(int) PIX_H / 2; i--)
	{	
		image_j = 0;
		for(int j = (int) PIX_W / 2; j >= -(int) PIX_W / 2; j--)
		{
			// update point coordinates:
			float p_x = nc_x + (up_x * i * Hnear / (1.0*PIX_H)) - 
						(right_x * j * Wnear / (1.0*PIX_W));
			float p_y = nc_y + (up_y * i * Hnear / (1.0*PIX_H)) - 
						(right_y * j * Wnear / (1.0*PIX_W));
			float p_z = nc_z + (up_z * i * Hnear / (1.0*PIX_H)) - 
						(right_z * j * Wnear / (1.0*PIX_W));

			// using the current p and the viewpoint we compute the parametric
			// equation of the ray
			// direction of the ray -- normalized
			float n_x = p_x - x_pos;
			float n_y = p_y - y_pos;
			float n_z = p_z - z_pos;
			float n_len = sqrt(n_x * n_x + n_y * n_y + n_z * n_z);
			n_x = n_x / n_len;
			n_y = n_y / n_len;
			n_z = n_z / n_len;

			fColorRGBA fcol = computeRayTrace(x_pos, y_pos, z_pos,
				n_x, n_y, n_z, 0);
			// fill color
			uColorRGBA ucol = transformColor(fcol);
			image[image_i][image_j][0] = ucol[0];
			image[image_i][image_j][0] = ucol[0];
			image[image_i][image_j][0] = ucol[0];
			image[image_i][image_j][0] = 255;
			image_j++;
		}
		image_i++;
	}
	std::string outputFilepath = "../output/";
	std::string outputImageName = "out_"+currentDateTime();
	unsigned int cols = PIX_W + 1;
	unsigned int rows = PIX_H + 1;
	encodePNG(image, cols, rows, outputFilepath + outputImageName);
}

void changeSize(int w, int h)
{
	// Prevent a divide by zero , when window is too short
	// you can't divide by zero
	if(h == 0)
		h = 1;

	perspective_ratio = 1.0*w / h;

	// reset the coordinate system before modifying
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	// set the viewport to be the entire window
	glViewport(0, 0, w, h);

	// set the correct perspective
	gluPerspective(perspective_angle, perspective_ratio, perspective_near,
				perspective_far); 
	glutPostRedisplay();
}

void processKeys(unsigned char key, int x, int y)
{
	switch(key)
	{	
		case 'w': 
			theta += 0.1;
			break;
		case 's':
			theta -= 0.1;
			break;
		case 'a':
			phi += 0.1;
			break;
		case 'd':
			phi -= 0.1;
			break;
		case 't': 
			if (R >= 2.25) R -= 0.25; 
			break;
		case 'g': 
			if (R < 100) R += 0.25; 
			break;
		case 'r':
			computeRayTraceWrapper();
			break;
		case 27:
			exit(0);
	}

	// we need to redisplay using the new position of the camera
	glutPostRedisplay();
}

Object createObject(const cv::Mat& points, const cv::Mat& faces)
{
	Object obj;

	unsigned long j = 0;
	std::cout << "faces.cols: " << faces.cols << "\n";
	while(j < (unsigned long) faces.cols)
	{
		if(j%4 != 0)
		{
			Triangle t;
			t.p.x = points.at<float>(faces.at<unsigned int>(0,j),0);
			t.p.y = points.at<float>(faces.at<unsigned int>(0,j),1);
			t.p.z = points.at<float>(faces.at<unsigned int>(0,j),2);
			t.p.index = faces.at<unsigned int>(0,j);
			j++;
			t.q.x = points.at<float>(faces.at<unsigned int>(0,j),0);
			t.q.y = points.at<float>(faces.at<unsigned int>(0,j),1);
			t.q.z = points.at<float>(faces.at<unsigned int>(0,j),2);
			t.q.index = faces.at<unsigned int>(0,j);
			j++;
			t.r.x = points.at<float>(faces.at<unsigned int>(0,j),0);
			t.r.y = points.at<float>(faces.at<unsigned int>(0,j),1);
			t.r.z = points.at<float>(faces.at<unsigned int>(0,j),2);
			t.r.index = faces.at<unsigned int>(0,j);
			j++;
			/* face normal for the given triangle */
			float qp_x = t.q.x - t.p.x;
			float qp_y = t.q.y - t.p.y;
			float qp_z = t.q.y - t.p.y;
			float rp_x = t.r.x - t.p.x;
			float rp_y = t.r.y - t.p.y;
			float rp_z = t.r.z - t.p.z;
			float norm_x = qp_y * rp_z - qp_z * rp_y;
			float norm_y = qp_z * rp_x - qp_x * rp_z;
			float norm_z = qp_x * rp_y - qp_y * rp_x;
			float norm_length = sqrt(norm_x*norm_x + norm_y*norm_y + norm_z*norm_z);
			norm_x = norm_x / norm_length;
			norm_y = norm_y / norm_length;
			norm_z = norm_z / norm_length;
			t.face_n[0] = norm_x;
			t.face_n[1] = norm_y;
			t.face_n[2] = norm_z;
			/* add triangle to object */
			obj.push_back(t);
		} else {
			j++;
		}
	}
	return obj;
}

void computeVertexNormals(Object& obj, const cv::Mat& points ,const cv::Mat& faces)
{
	std::vector< std::vector<Vec3D> > table;
	table.resize(points.rows);
	for(unsigned int i = 0; i < obj.size(); i++)
	{
		Triangle t = obj[i];
		Vec3D v;
		v.x = t.face_n[0];
		v.y = t.face_n[1];
		v.z = t.face_n[2];
		table[t.p.index].push_back(v);
		table[t.q.index].push_back(v);
		table[t.r.index].push_back(v);
	}
	std::cout << "table size: " << table.size() << "\n";
	/* not all vertices in the list will have a normal,
	 * only those which are a vertex in a triangle mesh, the others' normals
	 * are just (0,0,0) by default in Point3D
	 */ 
	int counter = 0;
	for(unsigned int i = 0; i < table.size(); i++)
	{
		if(table[i].size() < 1)
		{
			counter++;
		}
	}
	std::cout << "vertices not in a triangle mesh: " << counter << "\n";
	/* compute normal at each vertex by summing them up */
	std::vector<Vec3D > norms;
	int counter2 = 0;;
	for(unsigned int i = 0; i < table.size(); i++)
	{
		Vec3D v;
		if(table[i].size() >= 1)
		{
			float n_x = 0, n_y = 0, n_z = 0;
			for(unsigned int j = 0; j < table[i].size(); j++)
			{
				n_x += table[i][j].x;
				n_y += table[i][j].y;
				n_z += table[i][j].z;
			}
			float n_length;
			n_length = sqrt(n_x * n_x + n_y * n_y + n_z * n_z);
			n_x = n_x / n_length;
			n_y = n_y / n_length;
			n_z = n_z / n_length;
			v.x = n_x; v.y = n_y; v.z = n_z;
			norms.push_back(v);
		} else {
			counter2++;
			v.x = v.y = v.z = 0;
			norms.push_back(v);
		}
	}
	if(counter != counter2)
	{
		std::cerr << "the number of vertices without normals do not match!\n";
		exit(1);
	}
	if(norms.size() != (unsigned int) points.rows)
	{
		std::cerr << "Every point should a normal, including the default one!\n";
		exit(1);
	}
	/* store in obj the normal for each vertex */
	for(unsigned int i = 0; i < obj.size(); i++)
	{
		obj[i].p.n[0] = norms[obj[i].p.index].x;
		obj[i].p.n[1] = norms[obj[i].p.index].y;
		obj[i].p.n[2] = norms[obj[i].p.index].z;
		obj[i].q.n[0] = norms[obj[i].q.index].x;
		obj[i].q.n[1] = norms[obj[i].q.index].y;
		obj[i].q.n[2] = norms[obj[i].q.index].z;
		obj[i].r.n[0] = norms[obj[i].r.index].x;
		obj[i].r.n[1] = norms[obj[i].r.index].y;
		obj[i].r.n[2] = norms[obj[i].r.index].z;
	}
}

DataObject readPLY(const std::string& ply_file, bool ifNormals, bool ifColor)
{
	//int result;
	std::string ply_filename = ply_file;
	/* the following ply reader from opencv is rubbish 
	 * if vertex is given by only 3d position is fine
	 * but as in the case of bun_zipper.ply the vertex
	 * is given by 3d position + confidence + intensity 
	 * and it can't handle those options -- it is overwritten below
	 */
	cv::Mat points = cv::ppf_match_3d::loadPLYSimple(ply_filename.c_str(), 0);
	if(points.empty())
	{
		std::cerr << "Ply file couldn't be loaded: " << ply_filename << "\n";
		exit(1);
	}
	cv::viz::Mesh mesh = cv::viz::Mesh::load(ply_filename, cv::viz::Mesh::LOAD_PLY);	
	if(mesh.cloud.empty())
	{
		std::cerr << "Ply file could be loaded by Mesh::load: \n" << ply_filename << "\n";
	}

	PLYModel plymodel(ply_filename.c_str(), ifNormals, ifColor);
	std::cout << "ply_model: " << plymodel.positions.size() << "\n";
	if(plymodel.positions.size() != (unsigned int) points.rows)
	{
		std::cerr << "dimensions in reading don't match!\n";
		exit(1);
	}
	// points overwritten
	for(unsigned int i = 0; i < (unsigned int) points.rows; i++)
	{
		points.at<float>(i,0) = plymodel.positions[i].x;
		points.at<float>(i,1) = plymodel.positions[i].y;
		points.at<float>(i,2) = plymodel.positions[i].z;
	}
	// if the next viz block is uncommented it will show the mesh visualization
	/*	
	cv::theRNG().fill(mesh.colors, cv::RNG::UNIFORM, 0, 255);
	cv::Affine3d pose = cv::Affine3d().rotate(cv::Vec3d(0, 0.8, 0));

	cv::viz::Viz3d viz("show_mesh");
	viz.showWidget("coosys", cv::viz::WCoordinateSystem());
	viz.showWidget("mesh", cv::viz::WMesh(mesh), pose);
	viz.setRenderingProperty("mesh", cv::viz::SHADING, cv::viz::SHADING_PHONG);
	viz.showWidget("text2d", cv::viz::WText("Just mesh", cv::Point(20, 20), 20, cv::viz::Color::green()));
	viz.spin();*/
	/* the next part doesn't work as in the normals returned produce -nan values */
/*	cv::Mat normals(points.rows, 6, CV_32F);
	double viewpoint[3] = {x_pos, y_pos, z_pos};
	result = cv::ppf_match_3d::computeNormalsPC3d(points, normals, 6, false,
		viewpoint);
	if(!result)
	{
		std::cout << "could not compute normals\n";
		exit(1);
	}

	cv::Mat norms2;
	cv::viz::computeNormals(mesh, mesh.normals);
	norms2 = mesh.normals;*/
	/*****************************************************************************/

	cv::Mat faces = mesh.polygons;
	Object obj = createObject(points, faces);
	std::cout << "obj size: " << obj.size() << "\n";
	computeVertexNormals(obj, points, faces);
	DataObject d_obj;
	d_obj.obj = obj;
	d_obj.faces = faces;
	d_obj.points = points;
	return d_obj;
}

void drawObject(const Object& obj)
{
	for(unsigned int i = 0; i < obj.size(); i++)
	{
		Triangle t = obj[i];
		Point3D p = t.p;
		Point3D q = t.q;
		Point3D r = t.r;

		glNormal3f(p.n[0], p.n[1], p.n[2]);
		glVertex3f(p.x, p.y, p.z);
		glNormal3f(q.n[0], q.n[1], q.n[2]);
		glVertex3f(q.x, q.y, q.z);
		glNormal3f(r.n[0], r.n[1], r.n[2]);
		glVertex3f(r.x, r.y, r.z);
	}
}

void updateScaleTranslate(DataObject& d_obj, float sx, float sy, float sz,
	float tx, float ty, float tz)
{
	// scaling
	for(unsigned long i = 0; i < (unsigned long) d_obj.points.rows; i++)
	{
		d_obj.points.at<float>(i,0) *= sx;
		d_obj.points.at<float>(i,1) *= sy;
		d_obj.points.at<float>(i,2) *= sz;
	}
	
	// translation
	for(unsigned long i = 0; i < (unsigned long) d_obj.points.rows; i++)
	{
		d_obj.points.at<float>(i,0) += tx;
		d_obj.points.at<float>(i,1) += ty;
		d_obj.points.at<float>(i,2) += tz;
	}

	d_obj.obj = createObject(d_obj.points, d_obj.faces);
	computeVertexNormals(d_obj.obj, d_obj.points, d_obj.faces);
}

void display()
{
	x_pos = R*cos(phi)*cos(theta);
	y_pos = R*sin(theta);
	z_pos = R*sin(phi)*cos(theta);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// initialize the model view matrix with the identity;
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	// look from (x,y,z) to the origin of the coordinate system
	// construct the matrix and multiply it by the current modelview matrix
	
	glPushMatrix();
	if(cos(theta) > 0)
		up_y = 1.0;
	else
		up_y = -1.0;
	gluLookAt(x_pos, y_pos, z_pos, at_x, at_y, at_z, up_x, up_y , up_z);

	GLfloat	light_ambient[] = {Ia[0], Ia[1], Ia[2], Ia[3]}; 
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	
	if(light_extra != 0)
	{
		GLfloat light_position[] = {light_x / light_extra, light_y / light_extra,
		 light_z / light_extra, light_extra};
		glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	} else 
	{
		GLfloat light_position[] = {light_x, light_y, light_z, 0};
		glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	}

	GLfloat white_light[] = {Ip[0], Ip[1], Ip[2], Ip[3]};
	glLightfv(GL_LIGHT0, GL_DIFFUSE, white_light);
	glLightfv(GL_LIGHT0, GL_SPECULAR, white_light);
	
	GLfloat mat_specular[] = {ks[0], ks[1], ks[2], ks[3]};
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);  
	GLfloat mat_shininess[] = {s_exp};
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess); 
	GLfloat black[] = {ka[0], ka[1], ka[2], ka[3]};
	glMaterialfv(GL_FRONT,GL_AMBIENT, black);
	
	glPushMatrix();
	float sx = 5.0, sy = 5.0, sz = 5.0;
	float tx = 0.0, ty = 0.0, tz = 0.0;
	//equivalent to glScalef(5.0f,5.0f,5.0f);
	glBegin(GL_TRIANGLES);
	if(bunny1.ifAlreadyScaledTranslated == false)
	{
		updateScaleTranslate(bunny1, sx, sy, sz, tx, ty, tz);
		bunny1.ifAlreadyScaledTranslated = true;
	}
	drawObject(bunny1.obj);
	glEnd();
	glPopMatrix();

	glPushMatrix();
	sx = 5.0, sy = 5.0, sz = 5.0;
	tx = 0.5, ty = 0.0, tz = 0.5;
	//glTranslatef(tx, ty, tz);
	//glScalef(sx, sy, sz);
	glBegin(GL_TRIANGLES);
	if(bunny2.ifAlreadyScaledTranslated == false)
	{
		updateScaleTranslate(bunny2, sx, sy, sz, tx, ty, tz);
		bunny2.ifAlreadyScaledTranslated = true;
	}
	drawObject(bunny2.obj);
	glEnd();
	glPopMatrix();

	glPushMatrix();
	sx = 5.0, sy = 5.0, sz = 5.0;
	tx = -0.5, ty = 0.0, tz = 0.5;
	//glTranslatef(-0.5f, 0.0f, 0.5f);
	//glScalef(5.0f, 5.0f, 5.0f);
	glBegin(GL_TRIANGLES);
	if(buddha1.ifAlreadyScaledTranslated == false)
	{
		updateScaleTranslate(buddha1, sx, sy, sz, tx, ty, tz);
		buddha1.ifAlreadyScaledTranslated = true;
	}
	drawObject(buddha1.obj);
	glEnd();
	glPopMatrix();

	glPopMatrix();

	glutSwapBuffers();

	//Note that the 2 boxes do not change their relative position and get
	// drawn viewed from the "camera" position. That is because the camera's
	// matrix is loaded into the stack and afterwards the boxes are drawn
	// "on top" by pushing it, hence preserving the transformation of the camera.
}

void init()
{
	glClearColor(0,0,0,0);

	glEnable( GL_BLEND );
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	/* Set up illumination */
	glShadeModel(GL_SMOOTH);

	glEnable(GL_LIGHTING);

	glEnable(GL_LIGHT0);
	/***********************/

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_NORMALIZE);

	// setting up the projection matrix
	glMatrixMode(GL_PROJECTION);
	gluPerspective(perspective_angle, perspective_ratio, perspective_near, 
				perspective_far);
}

int main(int argc, char** argv)
{
	bunny1 = readPLY("../data/bunny/reconstruction/bun_zipper_res3.ply", false, false);
	bunny2 = readPLY("../data/bunny/reconstruction/bun_zipper_res3.ply", false, false);
	buddha1 = readPLY("../data/happy_buddha/happy_vrip_res4.ply", false, false);
	std::cout << "Total no of triangles: " << 
	bunny1.obj.size() + bunny2.obj.size() + buddha1.obj.size() << "\n\n";


	// initialize GLUT
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH | GLUT_MULTISAMPLE);
	glutInitWindowSize(PIXELS_WIDTH, PIXELS_HEIGHT);
	glutInitWindowPosition(100,100);
	glutCreateWindow(title.c_str());

	init();

	// setting background color
	glClearColor(background[0], background[1], background[2], background[3]);
	glClear(GL_COLOR_BUFFER_BIT);

	// callback functions
	glutReshapeFunc(changeSize);
	glutDisplayFunc(display);
	glutKeyboardFunc(processKeys);

	glutMainLoop();
	return 0;
}