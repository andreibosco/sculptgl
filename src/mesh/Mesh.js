define([
  'misc/Utils',
  'mesh/meshData/MeshData',
  'math3d/Octree',
  'render/Render'
], function (Utils, MeshData, Octree, Render) {

  'use strict';

  function Mesh(gl) {
    this.meshData_ = new MeshData(this); // the mesh data
    this.octree_ = new Octree(this); // octree
    this.render_ = gl ? new Render(gl, this) : null; // octree
    this.id_ = Mesh.ID++; // useful id to retrieve a mesh (dynamic mesh, multires mesh, voxel mesh)

    // dirty local edited part of the mesh
    this.updateLocalEdit_ = false;
    this.localStackEditFaces_ = [];

    this.localVerticesXYZ_ = null;
    this.localColorsRGB_ = null;
    this.localMaterialsPBR_ = null;
    this.localNormalsXYZ_ = null;
    this.localTrianglesABC_ = null;
  }

  Mesh.ID = 0;
  Mesh.sortFunction = function (meshA, meshB) {
    // render transparent (back to front) after opaque (front to back) ones
    var aTr = meshA.isTransparent();
    var bTr = meshB.isTransparent();
    if (aTr && !bTr) return 1;
    if (!aTr && bTr) return -1;
    return (meshB.getDepth() - meshA.getDepth()) * (aTr && bTr ? 1.0 : -1.0);
  };

  Mesh.prototype = {
    getID: function () {
      return this.id_;
    },
    setID: function (id) {
      this.id_ = id;
    },
    getMeshData: function () {
      return this.meshData_;
    },
    getOctree: function () {
      return this.octree_;
    },
    getRender: function () {
      return this.render_;
    },
    setMeshData: function (data) {
      this.meshData_ = data;
    },
    setOctree: function (octree) {
      this.octree_ = octree;
    },
    setRender: function (render) {
      this.render_ = render;
    },
    setIsLocalEdit: function (bool) {
      this.isLocalEdit_ = bool;
      if (!bool)
        this.localStackEditFaces_.length = 0;
    },
    isLocalEdit: function () {
      return this.isLocalEdit_;
    },
    getRenderVertices: function () {
      if (this.isLocalEdit()) return this.localVerticesXYZ_;
      if (this.isUsingDrawArrays()) return this.getVerticesDrawArrays();
      return this.isUsingTexCoords() ? this.getVerticesTexCoord() : this.getVertices();
    },
    getRenderNormals: function () {
      if (this.isLocalEdit()) return this.localNormalsXYZ_;
      if (this.isUsingDrawArrays()) return this.getNormalsDrawArrays();
      return this.isUsingTexCoords() ? this.getNormalsTexCoord() : this.getNormals();
    },
    getRenderColors: function () {
      if (this.isLocalEdit()) return this.localColorsRGB_;
      if (this.isUsingDrawArrays()) return this.getColorsDrawArrays();
      return this.isUsingTexCoords() ? this.getColorsTexCoord() : this.getColors();
    },
    getRenderMaterials: function () {
      if (this.isLocalEdit()) return this.localMaterialsPBR_;
      if (this.isUsingDrawArrays()) return this.getMaterialsDrawArrays();
      return this.isUsingTexCoords() ? this.getMaterialsTexCoord() : this.getMaterials();
    },
    getRenderTexCoords: function () {
      return this.isUsingDrawArrays() ? this.getTexCoordsDrawArrays() : this.getTexCoords();
    },
    getRenderTriangles: function () {
      if (this.isLocalEdit()) return this.localTrianglesABC_;
      return this.isUsingTexCoords() ? this.getTrianglesTexCoord() : this.getTriangles();
    },
    getRenderNbTriangles: function () {
      if (this.isLocalEdit()) return this.localTrianglesABC_.length / 3;
      return this.getNbTriangles();
    },
    getRenderNbEdges: function () {
      return this.getNbEdges();
    },
    /** Initialize stuffs for the mesh */
    init: function (ignoreTransform) {
      this.initColorsAndMaterials();
      this.allocateArrays();
      this.initTopology();
      this.updateGeometry();
      this.updateDuplicateColorsAndMaterials();
      if (!ignoreTransform)
        this.scaleAndCenter();
    },
    /** Init topoloy stuffs */
    initTopology: function () {
      this.initFaceRings();
      this.initEdges();
      this.initVertexRings();
      this.initRenderTriangles();
    },
    /** Updates the mesh Geometry */
    updateGeometry: function (iFaces, iVerts) {
      this.updateFacesAabbAndNormal(iFaces);
      this.updateVerticesNormal(iVerts);
      this.updateOctree(iFaces);
      this.updateDuplicateGeometry(iVerts);
      this.updateFlatShading(iFaces);
      if (iFaces)
        this.localStackEditFaces_.push(iFaces);
    },
    renderLocalEdit: function (main) {
      // getUniqueEditFaces, createLocalEditTriangles and createLocalEditVertices
      // use the 'global' arraybuffer pool to avoid alloc (at least until the updateBuffer call)
      var iFaces = !main.useLocalEdit_ ? null : this.getUniqueEditFaces();
      if (!iFaces) {
        main.getSculpt().getCurrentTool().updateMeshBuffers();
        return false;
      }
      var mapVerts = this.createLocalEditTriangles(iFaces);
      this.createLocalEditVertices(mapVerts);

      this.setIsLocalEdit(true);
      this.updateIndexBuffer();
      this.updateGeometryBuffers();
      this.updateColorBuffer();
      this.updateMaterialBuffer();
      this.getShader().draw(this.getRender(), main);
      this.setIsLocalEdit(false);
      return true;
    },
    getUniqueEditFaces: function () {
      var stackFaces = this.localStackEditFaces_;
      if (stackFaces.length === 0)
        return;
      var nbFaces = this.getNbFaces();
      if (this.getNbTriangles() < 20000 && !this.getDynamicTopology)
        return;
      var poolFaces = new Uint32Array(Utils.getMemory(4 * nbFaces), 0, nbFaces);

      var tagFlag = ++Utils.TAG_FLAG;
      var ftf = this.getFacesTagFlags();

      var acc = 0;
      for (var i = 0, nbStack = stackFaces.length; i < nbStack; ++i) {
        var st = stackFaces[i];
        for (var j = 0, nb = st.length; j < nb; ++j) {
          var id = st[j];
          if (ftf[id] === tagFlag)
            continue;
          ftf[id] = tagFlag;
          poolFaces[acc++] = id;
        }
      }
      if (acc / nbFaces > 0.3)
        return;
      return poolFaces.subarray(0, acc);
    },
    createLocalEditTriangles: function (iFaces) {
      var nbFaces = iFaces.length;

      var fAr = this.getFaces();

      var tagFlag = ++Utils.TAG_FLAG;
      var vtf = this.getVerticesTagFlags();

      var nbVertices = this.getNbVertices();
      var buffer = Utils.getMemory(4 * (nbFaces * 7 + nbVertices));
      var mapVerts = new Uint32Array(buffer, 4 * nbFaces, nbVertices);
      var tris = new Uint32Array(buffer, 4 * (nbVertices + nbFaces), nbFaces * 6);
      var nbVerts = 0;
      var acc = 0;
      for (var i = 0; i < nbFaces; ++i) {
        var idf = iFaces[i] * 4;
        var iv1 = fAr[idf];
        var iv2 = fAr[idf + 1];
        var iv3 = fAr[idf + 2];
        var iv4 = fAr[idf + 3];
        var iTri = acc * 3;
        tris[iTri] = iv1;
        tris[iTri + 1] = iv2;
        tris[iTri + 2] = iv3;
        ++acc;
        if (vtf[iv1] !== tagFlag) {
          mapVerts[iv1] = nbVerts++;
          vtf[iv1] = tagFlag;
        }
        if (vtf[iv2] !== tagFlag) {
          mapVerts[iv2] = nbVerts++;
          vtf[iv2] = tagFlag;
        }
        if (vtf[iv3] !== tagFlag) {
          mapVerts[iv3] = nbVerts++;
          vtf[iv3] = tagFlag;
        }
        if (iv4 >= 0) {
          if (vtf[iv4] !== tagFlag) {
            mapVerts[iv4] = nbVerts++;
            vtf[iv4] = tagFlag;
          }
          iTri = acc * 3;
          tris[iTri] = iv1;
          tris[iTri + 1] = iv3;
          tris[iTri + 2] = iv4;
          ++acc;
        }
      }
      this.localTrianglesABC_ = tris = tris.subarray(0, acc * 3);

      var lenVerts = 3 * nbVerts;
      var offset = nbFaces + nbVertices + 3 * acc;
      buffer = Utils.getMemory((offset + lenVerts * 4) * 4);
      this.localVerticesXYZ_ = new Float32Array(buffer, 4 * offset, lenVerts);
      this.localNormalsXYZ_ = new Float32Array(buffer, 4 * (offset + lenVerts), lenVerts);
      this.localColorsRGB_ = new Float32Array(buffer, 4 * (offset + 2 * lenVerts), lenVerts);
      this.localMaterialsPBR_ = new Float32Array(buffer, 4 * (offset + 3 * lenVerts), lenVerts);
      return mapVerts;
    },
    createLocalEditVertices: function (mapVerts) {
      var tris = this.localTrianglesABC_;
      var lenTris = tris.length;
      var lv = this.localVerticesXYZ_;
      var ln = this.localNormalsXYZ_;
      var lc = this.localColorsRGB_;
      var lm = this.localMaterialsPBR_;

      var vAr = this.getVertices();
      var nAr = this.getNormals();
      var cAr = this.getColors();
      var mAr = this.getMaterials();

      var tagFlag = ++Utils.TAG_FLAG;
      var vtf = this.getVerticesTagFlags();
      for (var i = 0; i < lenTris; ++i) {
        var idv = tris[i];
        var mapv = tris[i] = mapVerts[idv];
        if (vtf[mapv] === tagFlag)
          continue;
        vtf[mapv] = tagFlag;
        mapv *= 3;
        idv *= 3;
        lv[mapv] = vAr[idv];
        lv[mapv + 1] = vAr[idv + 1];
        lv[mapv + 2] = vAr[idv + 2];

        ln[mapv] = nAr[idv];
        ln[mapv + 1] = nAr[idv + 1];
        ln[mapv + 2] = nAr[idv + 2];

        lc[mapv] = cAr[idv];
        lc[mapv + 1] = cAr[idv + 1];
        lc[mapv + 2] = cAr[idv + 2];

        lm[mapv] = mAr[idv];
        lm[mapv + 1] = mAr[idv + 1];
        lm[mapv + 2] = mAr[idv + 2];
      }
    },
    /** Allocate mesh resources */
    allocateArrays: function () {
      this.getIndexData().allocateArrays();
      this.getVertexData().allocateArrays();
      this.getTexCoordsData().allocateArrays();
      this.getOctree().allocateArrays();
    }
  };

  // Basically... Mesh is a proxy/interface of all the stuffs below

  Utils.makeProxy(MeshData, Mesh, function (proto) {
    return function () {
      return proto.apply(this.getMeshData(), arguments);
    };
  });

  Utils.makeProxy(Render, Mesh, function (proto) {
    return function () {
      return proto.apply(this.getRender(), arguments);
    };
  });

  Utils.makeProxy(Octree, Mesh, function (proto) {
    return function () {
      return proto.apply(this.getOctree(), arguments);
    };
  });

  return Mesh;
});