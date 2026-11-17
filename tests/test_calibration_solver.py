import cv2
import numpy as np
import pytest

from handtrack.processing._calibration_solver import (
    StereoResult,
    calibrate_intrinsics_robust,
    calibrate_stereo_robust,
    point_rmse,
    robust_error_threshold,
    solve_camera_pose_graph,
    stereo_view_error,
    validate_heldout_board_reconstruction,
)


def test_stereo_score_uses_both_views_of_a_small_distant_board():
    objects=np.zeros((20,3),np.float64)
    objects[:,:2]=np.mgrid[1:6,1:5].T.reshape(-1,2)*.04
    matrix=np.array([[1200.,0,640],[0,1200,512],[0,0,1]])
    distortion=np.zeros(5)
    rotation=np.eye(3)
    translation=np.array([[-1.5],[0.],[0.]])
    rvec=np.array([.03,.02,.01])
    tvec=np.array([-.1,-.08,3.])
    image_a,_=cv2.projectPoints(objects,rvec,tvec,matrix,distortion)
    image_b,_=cv2.projectPoints(objects,rvec,tvec+translation.ravel(),matrix,distortion)
    rng=np.random.default_rng(10)
    image_a+=rng.normal(0,.2,image_a.shape)
    image_b+=rng.normal(0,.2,image_b.shape)
    # The previous single-camera pose score exceeds the quality limit even
    # with the correct camera calibration and subpixel corner noise.
    _,single_rvec,single_tvec=cv2.solvePnP(objects,image_a,matrix,distortion)
    predicted_a,_=cv2.projectPoints(objects,single_rvec,single_tvec,matrix,distortion)
    predicted_b,_=cv2.projectPoints(objects,single_rvec,single_tvec+translation,matrix,distortion)
    old_error=np.sqrt(.5*(point_rmse(image_a,predicted_a)**2+point_rmse(image_b,predicted_b)**2))
    assert old_error>1.5
    error=stereo_view_error(objects,image_a,image_b,matrix,distortion,matrix,distortion,
                            rotation,translation)
    assert error<.4
    reverse=stereo_view_error(objects,image_b,image_a,matrix,distortion,matrix,distortion,
                              rotation,-translation)
    np.testing.assert_allclose(error,reverse,atol=1e-6)
    # Jointly fitting the board must not hide an incorrect stereo transform.
    wrong=stereo_view_error(objects,image_a,image_b,matrix,distortion,matrix,distortion,
                            rotation,translation+np.array([[0.],[.1],[0.]]))
    assert wrong>1.5


def _synthetic_intrinsic_observations(count=36):
    rng = np.random.default_rng(741)
    object_grid = np.zeros((8 * 6, 3), np.float32)
    object_grid[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2) * 0.04
    camera_matrix = np.array(
        [[910.0, 0.0, 638.0], [0.0, 900.0, 509.0], [0.0, 0.0, 1.0]]
    )
    dist_coeffs = np.array([[-0.12, 0.035, 0.001, -0.0008, -0.006]])
    object_points = []
    image_points = []
    for index in range(count):
        rvec = np.array(
            [
                0.35 * np.sin(index * 0.37),
                0.42 * np.cos(index * 0.29),
                0.15 * np.sin(index * 0.19),
            ]
        )
        tvec = np.array(
            [
                0.16 * np.sin(index * 0.51) - 0.12,
                0.12 * np.cos(index * 0.43) - 0.10,
                0.72 + 0.20 * (index % 5) / 4,
            ]
        )
        projected, _ = cv2.projectPoints(
            object_grid, rvec, tvec, camera_matrix, dist_coeffs
        )
        noise_scale = 5.0 if index == 10 else 0.12
        projected += rng.normal(0.0, noise_scale, projected.shape)
        object_points.append(object_grid.copy())
        image_points.append(projected.astype(np.float32))
    return object_points, image_points, camera_matrix


def test_point_rmse_is_pointwise_euclidean_rms():
    observed = np.zeros((2, 1, 2), dtype=np.float32)
    projected = np.array([[[3.0, 4.0]], [[0.0, 0.0]]], dtype=np.float32)

    assert point_rmse(observed, projected) == np.sqrt(12.5)


def test_robust_threshold_rejects_large_tail():
    threshold = robust_error_threshold([0.20, 0.21, 0.22, 0.23, 4.0])

    assert 0.23 < threshold < 1.0


def test_robust_intrinsics_reject_outlier_and_recover_focal_lengths():
    object_points, image_points, expected_matrix = _synthetic_intrinsic_observations()

    result = calibrate_intrinsics_robust(
        object_points,
        image_points,
        (1280, 1024),
        min_inliers=20,
    )

    assert 10 in result.outlier_indices
    assert result.rms < 0.4
    assert np.max(result.heldout_errors) < 0.5
    np.testing.assert_allclose(
        np.diag(result.camera_matrix)[:2],
        np.diag(expected_matrix)[:2],
        rtol=0.025,
    )
    assert result.std_intrinsics.size > 0


def _make_pair(camera_a, camera_b, rotations, translations):
    relative_rotation = rotations[camera_b] @ rotations[camera_a].T
    relative_translation = (
        translations[camera_b] - relative_rotation @ translations[camera_a]
    )
    return StereoResult(
        camera_a=camera_a,
        camera_b=camera_b,
        rotation=relative_rotation,
        translation=relative_translation,
        rms=0.25,
        per_view_errors=np.full(30, 0.25),
        heldout_errors=np.full(6, 0.25),
        inlier_indices=np.arange(30),
        outlier_indices=np.empty(0, dtype=np.int32),
        heldout_indices=np.arange(6),
    )


def test_pose_graph_recovers_all_camera_poses():
    rotations = [np.eye(3)]
    translations = [np.zeros((3, 1))]
    for rvec, translation in (
        ([0.02, -0.16, 0.01], [0.42, 0.01, 0.04]),
        ([-0.03, 0.19, -0.01], [-0.39, 0.03, 0.06]),
        ([0.14, 0.02, 0.01], [0.03, 0.35, 0.10]),
    ):
        rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
        rotations.append(rotation)
        translations.append(np.asarray(translation, dtype=np.float64).reshape(3, 1))
    pairs = [
        _make_pair(a, b, rotations, translations)
        for a in range(4)
        for b in range(a + 1, 4)
    ]

    solved_rotations, solved_translations, loop_rms = solve_camera_pose_graph(4, pairs)

    assert loop_rms < 1e-8
    for expected, actual in zip(rotations, solved_rotations):
        np.testing.assert_allclose(actual, expected, atol=1e-8)
    for expected, actual in zip(translations, solved_translations):
        np.testing.assert_allclose(actual, expected, atol=1e-8)


def test_heldout_board_reconstruction_recovers_shape_scale_and_plane():
    camera_matrix = np.array(
        [[850.0, 0.0, 640.0], [0.0, 850.0, 512.0], [0.0, 0.0, 1.0]]
    )
    camera_matrices = [camera_matrix.copy() for _ in range(3)]
    dist_coeffs = [np.zeros((1, 5)) for _ in range(3)]
    rotations = [np.eye(3) for _ in range(3)]
    translations = [
        np.array([[0.0], [0.0], [0.0]]),
        np.array([[-0.35], [0.0], [0.0]]),
        np.array([[0.0], [-0.30], [0.0]]),
    ]
    board_points = np.zeros((5 * 4, 3), dtype=np.float32)
    board_points[:, :2] = np.mgrid[0:5, 0:4].T.reshape(-1, 2) * 0.04
    all_corners = [[] for _ in range(3)]
    all_ids = [[] for _ in range(3)]
    ids = np.arange(len(board_points), dtype=np.int32).reshape(-1, 1)
    for view in range(4):
        board_rotation, _ = cv2.Rodrigues(
            np.array([0.05 * view, -0.04 * view, 0.02 * view])
        )
        world_points = (board_rotation @ board_points.T).T + np.array(
            [-0.08, -0.06, 0.8 + 0.05 * view]
        )
        for camera in range(3):
            projected, _ = cv2.projectPoints(
                world_points,
                np.zeros(3),
                translations[camera],
                camera_matrices[camera],
                dist_coeffs[camera],
            )
            all_corners[camera].append(projected.astype(np.float32))
            all_ids[camera].append(ids.copy())

    result = validate_heldout_board_reconstruction(
        all_corners,
        all_ids,
        board_points,
        camera_matrices,
        dist_coeffs,
        rotations,
        translations,
        [0, 1, 2, 3],
    )

    assert result.num_views == 4
    assert result.num_points == 80
    assert result.shape_rms_mm < 0.01
    assert result.scale_error_percent < 0.01
    assert result.planarity_rms_mm < 0.01


@pytest.mark.parametrize("coefficients", [8, 12, 14])
@pytest.mark.parametrize("mixed_models", [False, True])
def test_stereo_preserves_extended_lens_models_and_fixed_intrinsics(coefficients, mixed_models):
    # Equal radial numerator/denominator terms cancel exactly. A mono fit of
    # an undistorted camera can legitimately return this rational representation.
    # Dropping denominator terms silently changes the lens during stereo fitting.
    objects = np.zeros((30, 3), np.float32)
    objects[:, :2] = np.mgrid[0:6, 0:5].T.reshape(-1, 2)*.04
    matrix = np.array([[900., 0, 640], [0, 890, 512], [0, 0, 1.]])
    distortion = np.zeros(coefficients)
    distortion[[0, 1, 4]] = [.6, .05, .002]
    distortion[5:8] = [.6, .05, .002]
    if coefficients >= 12:
        distortion[8:12] = [.002, -.001, .001, .0002]
    if coefficients == 14:
        distortion[12:14] = [.01, -.015]
    distortion_a = np.zeros(5) if mixed_models else distortion.copy()
    expected_rotation, _ = cv2.Rodrigues(np.array([.01, -.03, .005]))
    expected_translation = np.array([[-.2], [.005], [.01]])
    obj, first, second = [], [], []
    for index in range(20):
        rvec = np.array([.3*np.sin(index*.3), .3*np.cos(index*.4), .1*np.sin(index*.2)])
        position = np.array([-.1+.1*np.sin(index*.5), -.08+.1*np.cos(index*.3), .8+.2*(index%5)/4])
        rotation, _ = cv2.Rodrigues(rvec)
        second_rvec, _ = cv2.Rodrigues(expected_rotation@rotation)
        second_position = expected_rotation@position.reshape(3, 1)+expected_translation
        image_a, _ = cv2.projectPoints(objects, rvec, position, matrix, distortion_a)
        image_b, _ = cv2.projectPoints(objects, second_rvec, second_position, matrix, distortion)
        obj.append(objects.copy())
        first.append(image_a.astype(np.float32))
        second.append(image_b.astype(np.float32))
    before = [array.copy() for array in (matrix, distortion_a, distortion)]
    result = calibrate_stereo_robust(0, 1, obj, first, second, matrix, distortion_a,
                                     matrix, distortion, (1280, 1024), min_inliers=8)
    np.testing.assert_allclose(result.translation, expected_translation, atol=1e-4)
    np.testing.assert_allclose(result.rotation, expected_rotation, atol=1e-4)
    assert result.rms < .001
    assert np.max(result.heldout_errors) < .001
    for actual, original in zip((matrix, distortion_a, distortion), before):
        np.testing.assert_array_equal(actual, original)
